import json
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .knn_retriever import KNNGroundwaterRetriever
from .keyword_baseline import KeywordBaselineMatcher
from .retrieval_pipeline import GroundwaterRetrievalPipeline

class RetrievalEvaluator:
    """
    Evaluator for ranking quality (Precision@k, MRR) and semantic robustness across
    differently phrased queries, comparing KNN pipeline against the keyword baseline.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.pipeline = GroundwaterRetrievalPipeline(data_dir=data_dir)
        self.knn_retriever = self.pipeline.retriever
        self.baseline_matcher = self.pipeline.baseline
        
        benchmark_path = os.path.join(data_dir, "benchmark_test_queries.json")
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.test_groups = json.load(f)

    def evaluate_query(self, query: str, true_intent: str, true_site_no: str, k_list: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluates a single query on both KNN and Keyword baseline across:
        1. Category alignment Precision@k (does the retrieved neighbor match the intent?)
        2. Structured-data lookup Precision@k (does the ranked lookup match intent AND site?)
        """
        max_k = max(k_list)
        # Run KNN retriever
        knn_res = self.knn_retriever.retrieve(query, k=max_k)
        knn_lookups = self.pipeline.rank_structured_lookups(
            query,
            knn_res.get("ranked_intents", [knn_res["predicted_intent"]]),
            knn_res.get("entities", {}),
            knn_res["top_k_matches"],
            k=max_k,
        )

        # Run Baseline
        base_res = self.baseline_matcher.retrieve(query, k=max_k)
        base_lookups = self.pipeline.rank_structured_lookups(
            query,
            base_res.get("ranked_intents", [base_res["predicted_intent"]]),
            base_res.get("entities", {}),
            base_res["top_k_matches"],
            k=max_k,
        )
        
        results = {}
        for model_name, res, lookups in [
            ("knn", knn_res, knn_lookups),
            ("baseline", base_res, base_lookups),
        ]:
            matches = res["top_k_matches"]
            intent_correct = (res["predicted_intent"] == true_intent)
            
            cat_p_at_k = {}
            rec_p_at_k = {}
            for k in k_list:
                top_slice = matches[:k]
                if not top_slice:
                    cat_p_at_k[f"P@{k}"] = 0.0
                else:
                    cat_relevant = sum(1 for item in top_slice if item["intent"] == true_intent)
                    cat_p_at_k[f"P@{k}"] = round(cat_relevant / k, 4)
                
                # Structured lookup relevance: ranked (site_no, intent) lookup matches target
                lookup_slice = lookups[:k]
                if not lookup_slice:
                    rec_p_at_k[f"P@{k}"] = 0.0
                else:
                    rec_relevant = sum(
                        1 for item in lookup_slice
                        if item["intent"] == true_intent and item.get("site_no") == true_site_no
                    )
                    rec_p_at_k[f"P@{k}"] = round(rec_relevant / k, 4)
                
            # Reciprocal rank for category
            cat_rr = 0.0
            for rank_idx, item in enumerate(matches, 1):
                if item["intent"] == true_intent:
                    cat_rr = round(1.0 / rank_idx, 4)
                    break

            # Reciprocal rank for structured lookup record
            rec_rr = 0.0
            for rank_idx, item in enumerate(lookups, 1):
                if item["intent"] == true_intent and item.get("site_no") == true_site_no:
                    rec_rr = round(1.0 / rank_idx, 4)
                    break

            top_site = lookups[0]["site_no"] if lookups else None
            results[model_name] = {
                "predicted_intent": res["predicted_intent"],
                "resolved_site_no": top_site,
                "intent_correct": intent_correct,
                "category_precision_at_k": cat_p_at_k,
                "record_precision_at_k": rec_p_at_k,
                "category_mrr": cat_rr,
                "record_mrr": rec_rr
            }
            
        return results

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation across all test groups and paraphrased queries.
        Returns aggregate metrics comparing KNN vs Keyword matching.
        """
        knn_cat_p1, knn_cat_p3, knn_cat_p5, knn_cat_mrr = [], [], [], []
        knn_rec_p1, knn_rec_p3, knn_rec_p5, knn_rec_mrr = [], [], [], []
        knn_intent_acc = []
        
        base_cat_p1, base_cat_p3, base_cat_p5, base_cat_mrr = [], [], [], []
        base_rec_p1, base_rec_p3, base_rec_p5, base_rec_mrr = [], [], [], []
        base_intent_acc = []
        
        group_summaries = []
        total_queries = 0

        for group in self.test_groups:
            group_name = group["test_group"]
            target_category = group["category"]
            target_site = group["target_site_no"]
            phrasings = group["phrasings"]
            
            group_evals = []
            for query in phrasings:
                total_queries += 1
                ev = self.evaluate_query(query, target_category, target_site, k_list=[1, 3, 5])
                
                # KNN metrics
                knn_ev = ev["knn"]
                knn_cat_p1.append(knn_ev["category_precision_at_k"]["P@1"])
                knn_cat_p3.append(knn_ev["category_precision_at_k"]["P@3"])
                knn_cat_p5.append(knn_ev["category_precision_at_k"]["P@5"])
                knn_cat_mrr.append(knn_ev["category_mrr"])
                knn_rec_p1.append(knn_ev["record_precision_at_k"]["P@1"])
                knn_rec_p3.append(knn_ev["record_precision_at_k"]["P@3"])
                knn_rec_p5.append(knn_ev["record_precision_at_k"]["P@5"])
                knn_rec_mrr.append(knn_ev["record_mrr"])
                knn_intent_acc.append(1 if knn_ev["intent_correct"] else 0)
                
                # Baseline metrics
                base_ev = ev["baseline"]
                base_cat_p1.append(base_ev["category_precision_at_k"]["P@1"])
                base_cat_p3.append(base_ev["category_precision_at_k"]["P@3"])
                base_cat_p5.append(base_ev["category_precision_at_k"]["P@5"])
                base_cat_mrr.append(base_ev["category_mrr"])
                base_rec_p1.append(base_ev["record_precision_at_k"]["P@1"])
                base_rec_p3.append(base_ev["record_precision_at_k"]["P@3"])
                base_rec_p5.append(base_ev["record_precision_at_k"]["P@5"])
                base_rec_mrr.append(base_ev["record_mrr"])
                base_intent_acc.append(1 if base_ev["intent_correct"] else 0)
                
                group_evals.append({
                    "query": query,
                    "knn_intent": knn_ev["predicted_intent"],
                    "knn_cat_p1": knn_ev["category_precision_at_k"]["P@1"],
                    "knn_rec_p1": knn_ev["record_precision_at_k"]["P@1"],
                    "resolved_site": knn_ev["resolved_site_no"],
                    "base_intent": base_ev["predicted_intent"],
                    "base_cat_p1": base_ev["category_precision_at_k"]["P@1"],
                    "base_rec_p1": base_ev["record_precision_at_k"]["P@1"],
                    "manual_check": "PASS" if (knn_ev["intent_correct"] and knn_ev["resolved_site_no"] == target_site) else "REVIEW",
                })
                
            group_pass_rate = round(
                sum(1 for e in group_evals if e["manual_check"] == "PASS") / max(1, len(group_evals)),
                4,
            )
            group_summaries.append({
                "group_name": group_name,
                "category": target_category,
                "target_site": target_site,
                "phrasings_count": len(phrasings),
                "manual_check_pass_rate": group_pass_rate,
                "queries_evaluated": group_evals
            })

        overall_metrics = {
            "total_test_queries": total_queries,
            "knn": {
                "intent_accuracy": round(float(np.mean(knn_intent_acc)), 4),
                "category_retrieval": {
                    "mean_p_at_1": round(float(np.mean(knn_cat_p1)), 4),
                    "mean_p_at_3": round(float(np.mean(knn_cat_p3)), 4),
                    "mean_p_at_5": round(float(np.mean(knn_cat_p5)), 4),
                    "mrr": round(float(np.mean(knn_cat_mrr)), 4)
                },
                "exact_record_retrieval": {
                    "mean_p_at_1": round(float(np.mean(knn_rec_p1)), 4),
                    "mean_p_at_3": round(float(np.mean(knn_rec_p3)), 4),
                    "mean_p_at_5": round(float(np.mean(knn_rec_p5)), 4),
                    "mrr": round(float(np.mean(knn_rec_mrr)), 4)
                }
            },
            "keyword_baseline": {
                "intent_accuracy": round(float(np.mean(base_intent_acc)), 4),
                "category_retrieval": {
                    "mean_p_at_1": round(float(np.mean(base_cat_p1)), 4),
                    "mean_p_at_3": round(float(np.mean(base_cat_p3)), 4),
                    "mean_p_at_5": round(float(np.mean(base_cat_p5)), 4),
                    "mrr": round(float(np.mean(base_cat_mrr)), 4)
                },
                "exact_record_retrieval": {
                    "mean_p_at_1": round(float(np.mean(base_rec_p1)), 4),
                    "mean_p_at_3": round(float(np.mean(base_rec_p3)), 4),
                    "mean_p_at_5": round(float(np.mean(base_rec_p5)), 4),
                    "mrr": round(float(np.mean(base_rec_mrr)), 4)
                }
            },
            "group_summaries": group_summaries
        }
        
        return overall_metrics
