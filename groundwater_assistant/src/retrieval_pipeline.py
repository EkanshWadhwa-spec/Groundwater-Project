import os
from typing import Dict, Any, Optional

from .data_loader import USGSDataLoader
from .preprocessor import QueryPreprocessor
from .knn_retriever import KNNGroundwaterRetriever
from .keyword_baseline import KeywordBaselineMatcher

class GroundwaterRetrievalPipeline:
    """
    End-to-end pipeline connecting natural language query normalization,
    TF-IDF KNN category classification, entity extraction, and structured USGS NWIS data retrieval,
    with comprehensive multi-metric location dossiers (groundwater depth, soil moisture, trend, quality).
    """
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        self.data_dir = data_dir
        
        self.data_loader = USGSDataLoader(data_dir=data_dir)
        self.preprocessor = QueryPreprocessor()
        self.retriever = KNNGroundwaterRetriever(data_dir=data_dir)
        self.baseline = KeywordBaselineMatcher(data_dir=data_dir)

    def rank_structured_lookups(
        self,
        raw_query: str,
        ranked_intents: list,
        entities: Dict[str, Any],
        neighbor_matches: list,
        k: int = 5,
    ) -> list:
        """
        Produces top-k ranked structured-data lookup candidates (site_no + category)
        by combining resolved target station(s) with ranked category intents.
        """
        candidate_sites = self.data_loader.find_matching_sites(entities, raw_query)
        if not candidate_sites and neighbor_matches:
            seen = set()
            for m in neighbor_matches:
                sno = m.get("target_site_no")
                if sno and sno not in seen:
                    s = self.data_loader.get_site(sno)
                    if s:
                        candidate_sites.append(s)
                        seen.add(sno)
        if not candidate_sites:
            candidate_sites = [self.data_loader.get_all_sites().iloc[0].to_dict()]

        lookups = []
        seen_pairs = set()
        # Rank primary site across ranked intents first, then secondary matching sites
        for site in candidate_sites:
            for intent in ranked_intents:
                pair = (site["site_no"], intent)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    lookups.append({
                        "site_no": site["site_no"],
                        "station_nm": site["station_nm"],
                        "intent": intent,
                        "lookup_key": f"{site['site_no']}::{intent}",
                    })
                if len(lookups) >= k:
                    return lookups
        return lookups[:k]

    def process_query(self, raw_query: str, k: int = 5) -> Dict[str, Any]:
        """
        Processes a field-worker natural language query and returns the structured response
        along with the complete location hydro-soil dossier.
        """
        # 1. Normalization & Entity Extraction
        entities = self.preprocessor.extract_entities(raw_query)
        norm_query = self.preprocessor.normalize(raw_query)
        
        # 2. KNN Retrieval & Classification
        knn_res = self.retriever.retrieve(raw_query, k=k)
        predicted_intent = knn_res["predicted_intent"]
        confidence = knn_res["confidence"]
        
        # 3. Keyword Baseline classification for comparison
        baseline_res = self.baseline.retrieve(raw_query, k=k)
        
        # 4. Resolve Target Site
        resolved_site = None
        if entities.get("site_no"):
            resolved_site = self.data_loader.get_site(entities["site_no"])
            
        if not resolved_site and (entities.get("county") or entities.get("state") or entities.get("aquifer")):
            resolved_site = self.data_loader.find_site_by_entities(entities)
            
        # Free-form place search fallback (strictly no blind default)
        if not resolved_site:
            resolved_site = self.data_loader.search_by_place_name(raw_query, allow_default=False)
            
        # Fallback to top KNN neighbor's target site if query had no explicit entity
        if not resolved_site and knn_res["top_k_matches"]:
            top_neighbor_site = knn_res["top_k_matches"][0].get("target_site_no")
            if top_neighbor_site:
                resolved_site = self.data_loader.get_site(top_neighbor_site)
                
        # Ultimate fallback
        if not resolved_site:
            resolved_site = self.data_loader.get_all_sites().iloc[0].to_dict()
            
        site_no = resolved_site["site_no"]
        
        # 5. Retrieve Structured Data Record from USGS Dataset
        structured_record = self.data_loader.get_structured_record(site_no, predicted_intent)
        
        # 6. Retrieve Complete Hydro-Soil Dossier (Groundwater Level, Soil Moisture, Trend, Quality)
        location_dossier = self.data_loader.get_full_location_dossier(site_no)

        # 7. Ranked structured lookups & field-worker decision takeaway
        top_k_lookups = self.rank_structured_lookups(
            raw_query,
            knn_res.get("ranked_intents", [predicted_intent]),
            entities,
            knn_res["top_k_matches"],
            k=k,
        )
        decision_summary = self.data_loader.build_decision_summary(
            resolved_site, predicted_intent, structured_record, location_dossier
        )
        
        return {
            "query": raw_query,
            "normalized_query": norm_query,
            "intent_normalized_query": knn_res.get("intent_normalized_query", norm_query),
            "tfidf_terms": knn_res.get("tfidf_terms", []),
            "entities": entities,
            "intent": predicted_intent,
            "confidence": confidence,
            "intent_probabilities": knn_res["intent_probabilities"],
            "top_k_neighbors": knn_res["top_k_matches"],
            "top_k_lookups": top_k_lookups,
            "baseline_intent": baseline_res["predicted_intent"],
            "baseline_confidence": baseline_res["confidence"],
            "baseline_matches": baseline_res["top_k_matches"],
            "target_site": resolved_site,
            "structured_record": structured_record,
            "location_dossier": location_dossier,
            "decision_summary": decision_summary,
        }
