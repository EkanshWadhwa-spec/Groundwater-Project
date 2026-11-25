import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from .preprocessor import QueryPreprocessor

class KNNGroundwaterRetriever:
    """
    K-Nearest Neighbors and TF-IDF retrieval pipeline for groundwater query classification
    and structured-data lookup matching.
    """
    def __init__(self, data_dir: Optional[str] = None, n_neighbors: int = 5):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        self.data_dir = data_dir
        self.n_neighbors = n_neighbors
        self.preprocessor = QueryPreprocessor()
        
        # Load training queries
        train_path = os.path.join(data_dir, "training_queries.csv")
        self.train_df = pd.read_csv(train_path, dtype={"target_site_no": str})
        self.train_df["normalized_text"] = self.train_df["query_text"].apply(self.preprocessor.normalize)
        self.train_df["intent_normalized_text"] = self.train_df["query_text"].apply(self.preprocessor.normalize_for_intent)
        
        # Build TF-IDF vectorizer (Unigrams and Bigrams) on entity-disentangled tokens
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.train_df["intent_normalized_text"])
        
        # Fit NearestNeighbors model with cosine distance
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine", algorithm="brute")
        self.knn.fit(self.tfidf_matrix)
        
        # Also fit a KNeighborsClassifier for direct class probability estimation
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="cosine", weights="distance")
        self.clf.fit(self.tfidf_matrix, self.train_df["intent"])
        
        self.classes_ = list(self.clf.classes_)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def explain_query_tfidf(self, query: str, top_n: int = 8) -> List[Dict[str, Any]]:
        """
        Returns the top-weighted TF-IDF unigram/bigram features for a given query.
        """
        intent_norm = self.preprocessor.normalize_for_intent(query)
        vec = self.vectorizer.transform([intent_norm])
        if vec.nnz == 0:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        indices = vec.nonzero()[1]
        weights = [(feature_names[idx], round(float(vec[0, idx]), 4)) for idx in indices]
        weights.sort(key=lambda x: x[1], reverse=True)
        return [{"term": t, "tfidf_weight": w} for t, w in weights[:top_n]]

    def retrieve(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Classifies query intent and retrieves top-k nearest matching queries via TF-IDF + KNN.
        """
        if k is None:
            k = self.n_neighbors
            
        norm_query = self.preprocessor.normalize(query)
        intent_norm_query = self.preprocessor.normalize_for_intent(query)
        entities = self.preprocessor.extract_entities(query)
        
        query_vec = self.vectorizer.transform([intent_norm_query])
        
        # If query vector has all zeros (no known vocabulary), handle gracefully
        if query_vec.nnz == 0:
            return {
                "query": query,
                "normalized_query": norm_query,
                "intent_normalized_query": intent_norm_query,
                "predicted_intent": "location",
                "confidence": 0.25,
                "intent_probabilities": {c: 0.25 for c in self.classes_},
                "ranked_intents": list(self.classes_),
                "tfidf_terms": [],
                "entities": entities,
                "top_k_matches": []
            }
            
        # Get distances and indices from NearestNeighbors
        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=min(k, len(self.train_df)))
        
        # Get predicted probabilities
        probs = self.clf.predict_proba(query_vec)[0]
        prob_dict = {cls_name: float(round(p, 4)) for cls_name, p in zip(self.classes_, probs)}
        ranked_intents = [cls_name for cls_name, _ in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)]
        predicted_intent = self.clf.predict(query_vec)[0]
        confidence = float(round(np.max(probs), 4))
        
        # Format top-k matches
        top_k = []
        for dist, idx in zip(distances[0], indices[0]):
            sim = max(0.0, 1.0 - dist)
            row = self.train_df.iloc[idx]
            top_k.append({
                "query_text": row["query_text"],
                "intent": row["intent"],
                "target_site_no": row["target_site_no"],
                "cosine_similarity": round(float(sim), 4),
                "distance": round(float(dist), 4)
            })
            
        return {
            "query": query,
            "normalized_query": norm_query,
            "intent_normalized_query": intent_norm_query,
            "predicted_intent": predicted_intent,
            "confidence": confidence,
            "intent_probabilities": prob_dict,
            "ranked_intents": ranked_intents,
            "tfidf_terms": self.explain_query_tfidf(query),
            "entities": entities,
            "top_k_matches": top_k
        }
