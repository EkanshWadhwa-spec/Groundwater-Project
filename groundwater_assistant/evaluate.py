import os
import sys
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from src.evaluator import RetrievalEvaluator

def main():
    data_dir = os.path.join(base_dir, "data")
    evaluator = RetrievalEvaluator(data_dir=data_dir)
    print("=" * 75)
    print(" GROUNDWATER QUERY CLASSIFICATION & RETRIEVAL ASSISTANT BENCHMARK ")
    print(" Dataset: USGS National Water Information System (NWIS)")
    print(" Pipeline: TF-IDF + Cosine Distance KNN vs Keyword-Matching Baseline")
    print("=" * 75)
    
    results = evaluator.run_benchmark()
    
    knn = results["knn"]
    base = results["keyword_baseline"]
    total = results["total_test_queries"]
    num_groups = len(results["group_summaries"])
    
    print(f"\nEvaluated {total} paraphrased, semantically identical test queries across {num_groups} domain groups.")
    
    print("\n" + "=" * 75)
    print(" 1. STRUCTURED CATEGORY RETRIEVAL (Location, Depth, Trend, Quality)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'TF-IDF + KNN':<18} | {'Keyword Baseline':<16}")
    print("-" * 75)
    print(f"{'Intent Accuracy':<32} | {knn['intent_accuracy'] * 100:.1f}%{'':<12} | {base['intent_accuracy'] * 100:.1f}%")
    print(f"{'Category Precision @ 1 (P@1)':<32} | {knn['category_retrieval']['mean_p_at_1']:.4f}{'':<12} | {base['category_retrieval']['mean_p_at_1']:.4f}")
    print(f"{'Category Precision @ 3 (P@3)':<32} | {knn['category_retrieval']['mean_p_at_3']:.4f}{'':<12} | {base['category_retrieval']['mean_p_at_3']:.4f}")
    print(f"{'Category Precision @ 5 (P@5)':<32} | {knn['category_retrieval']['mean_p_at_5']:.4f}{'':<12} | {base['category_retrieval']['mean_p_at_5']:.4f}")
    print(f"{'Category MRR':<32} | {knn['category_retrieval']['mrr']:.4f}{'':<12} | {base['category_retrieval']['mrr']:.4f}")

    print("\n" + "=" * 75)
    print(" 2. EXACT RECORD RETRIEVAL (Category + Site Matching)")
    print("=" * 75)
    print(f"{'Metric':<32} | {'TF-IDF + KNN':<18} | {'Keyword Baseline':<16}")
    print("-" * 75)
    print(f"{'Record Precision @ 1 (P@1)':<32} | {knn['exact_record_retrieval']['mean_p_at_1']:.4f}{'':<12} | {base['exact_record_retrieval']['mean_p_at_1']:.4f}")
    print(f"{'Record Precision @ 3 (P@3)':<32} | {knn['exact_record_retrieval']['mean_p_at_3']:.4f}{'':<12} | {base['exact_record_retrieval']['mean_p_at_3']:.4f}")
    print(f"{'Record Precision @ 5 (P@5)':<32} | {knn['exact_record_retrieval']['mean_p_at_5']:.4f}{'':<12} | {base['exact_record_retrieval']['mean_p_at_5']:.4f}")
    print(f"{'Record MRR':<32} | {knn['exact_record_retrieval']['mrr']:.4f}{'':<12} | {base['exact_record_retrieval']['mrr']:.4f}")
    print("-" * 75)
    
    print("\n[SEMANTIC PARAPHRASE MANUAL RELEVANCE CHECK SUMMARY]")
    for g in results["group_summaries"]:
        pass_pct = g.get("manual_check_pass_rate", 0.0) * 100
        print(f"\n* {g['group_name']} [Expected: {g['category'].upper()}, Site: {g['target_site']}] — Pass Rate: {pass_pct:.0f}%")
        for q in g["queries_evaluated"][:2]:
            print(f"   Query: \"{q['query']}\"")
            print(
                f"     -> KNN: {q['knn_intent']} @ {q['resolved_site']} (Rec P@1: {q['knn_rec_p1']}) "
                f"| Baseline: {q['base_intent']} (Rec P@1: {q['base_rec_p1']}) | Status: {q['manual_check']}"
            )

    # Save output to evaluation_report.json
    report_path = os.path.join(base_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation artifact saved to: {report_path}")

if __name__ == "__main__":
    main()
