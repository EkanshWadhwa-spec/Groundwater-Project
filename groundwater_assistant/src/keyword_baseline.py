import re
from typing import Dict, Any, List, Optional
import pandas as pd

from .preprocessor import QueryPreprocessor

class KeywordBaselineMatcher:
    """
    Lexical keyword-matching baseline for query intent classification and record lookup.
    Used as an experimental benchmark against the TF-IDF + KNN retrieval pipeline.
    """
    def __init__(self, data_dir: Optional[str] = None):
        self.preprocessor = QueryPreprocessor()
        
        # Define strict lexical keyword dictionaries for categories
        self.intent_keywords = {
            "location": {
                "where", "location", "locate", "coordinates", "gps", "latitude", 
                "longitude", "lat", "long", "county", "state", "situated", "position", 
                "aquifer", "address", "placement", "map"
            },
            "depth": {
                "depth", "deep", "level", "feet", "ft", "table", "surface", 
                "down", "current", "latest", "present", "measurement", "height"
            },
            "trend": {
                "trend", "trends", "historical", "history", "time", "series", 
                "fluctuation", "fluctuations", "change", "changes", "drawdown", 
                "depletion", "falling", "rising", "declining", "recharge", 
                "hydrograph", "curve", "timeline", "months", "seasonal", "annual"
            },
            "quality": {
                "quality", "safe", "drinking", "potability", "potable", "ph", 
                "nitrate", "conductance", "conductivity", "salinity", "solids", 
                "mineral", "tds", "chemical", "contamination", "purity", 
                "alkaline", "acidic", "epa", "standards", "health"
            }
        }
        
        # Load training dataset for retrieval comparison
        if data_dir is not None:
            import os
            train_path = os.path.join(data_dir, "training_queries.csv")
            self.train_df = pd.read_csv(train_path, dtype={"target_site_no": str})
        else:
            self.train_df = None

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classifies intent based on exact lexical keyword matches.
        """
        norm_query = self.preprocessor.normalize(query)
        tokens = set(norm_query.split())
        
        scores = {}
        for intent, kw_set in self.intent_keywords.items():
            overlap = tokens.intersection(kw_set)
            scores[intent] = len(overlap)
            
        total_hits = sum(scores.values())
        if total_hits == 0:
            # Default fallback when no keywords match
            predicted_intent = "location"
            confidence = 0.25
            ranked_intents = ["location", "depth", "trend", "quality"]
        else:
            ranked_intents = [k for k, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
            predicted_intent = ranked_intents[0]
            confidence = round(scores[predicted_intent] / total_hits, 4)
            
        return {
            "query": query,
            "normalized_query": norm_query,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "ranked_intents": ranked_intents,
            "keyword_scores": scores,
            "entities": self.preprocessor.extract_entities(query)
        }

    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves top-k queries using Jaccard word-overlap lexical similarity.
        """
        base_result = self.classify_intent(query)
        query_tokens = set(base_result["normalized_query"].split())
        
        matches = []
        if self.train_df is not None:
            for _, row in self.train_df.iterrows():
                target_tokens = set(self.preprocessor.normalize(row["query_text"]).split())
                intersection = len(query_tokens.intersection(target_tokens))
                union = len(query_tokens.union(target_tokens))
                jaccard_sim = intersection / union if union > 0 else 0.0
                matches.append({
                    "query_text": row["query_text"],
                    "intent": row["intent"],
                    "target_site_no": row["target_site_no"],
                    "keyword_overlap": intersection,
                    "jaccard_similarity": round(jaccard_sim, 4)
                })
            # Sort descending by overlap and jaccard
            matches.sort(key=lambda x: (x["keyword_overlap"], x["jaccard_similarity"]), reverse=True)
            top_k = matches[:k]
        else:
            top_k = []
            
        base_result["top_k_matches"] = top_k
        return base_result
