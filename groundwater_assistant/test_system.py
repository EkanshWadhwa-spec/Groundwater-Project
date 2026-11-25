import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessor import QueryPreprocessor
from src.data_loader import USGSDataLoader
from src.knn_retriever import KNNGroundwaterRetriever
from src.keyword_baseline import KeywordBaselineMatcher
from src.retrieval_pipeline import GroundwaterRetrievalPipeline
from src.evaluator import RetrievalEvaluator
from src.usgs_nwis_fetcher import get_dataset_provenance, _parse_usgs_rdb

def run_tests():
    print(">>> Running Unit Tests for Groundwater Query Assistant...")
    
    # Test 1: Preprocessor
    prep = QueryPreprocessor()
    norm = prep.normalize("Where is well 01646500 located in Montgomery County?")
    intent_norm = prep.normalize_for_intent("Time series hydrograph for Arizona station 332450111584001")
    entities = prep.extract_entities("Where is well 01646500 located in Montgomery County?")
    assert "01646500" in norm, "Expected site_no preserved in general normalize()"
    assert "332450111584001" not in intent_norm and "arizona" not in intent_norm, f"Expected entities stripped in intent_norm: {intent_norm}"
    assert entities["site_no"] == "01646500", f"Expected 01646500, got {entities['site_no']}"
    assert entities["county"] == "Montgomery", f"Expected Montgomery, got {entities['county']}"
    print("Test 1 Passed: Preprocessor normalization, intent disentanglement & entity extraction.")
    
    # Test 2: Data Loader
    loader = USGSDataLoader(os.path.join(BASE_DIR, "data"))
    site = loader.get_site("01646500")
    assert site is not None, "Failed to retrieve site 01646500"
    ts = loader.get_timeseries("01646500")
    assert len(ts) > 0, "Failed to retrieve timeseries"
    qual = loader.get_quality("01646500")
    assert qual is not None, "Failed to retrieve water quality"
    assert loader.search_by_place_name("nonexistent_place_xyz", allow_default=False) is None
    print("Test 2 Passed: USGSDataLoader retrieves sites, timeseries, and water quality.")
    
    # Test 3: KNN Retriever
    retriever = KNNGroundwaterRetriever(os.path.join(BASE_DIR, "data"))
    res = retriever.retrieve("What is the current water level at site 01646500?")
    assert res["predicted_intent"] == "depth", f"Expected depth, got {res['predicted_intent']}"
    assert len(res["top_k_matches"]) > 0, "No top_k matches returned"
    assert len(res["tfidf_terms"]) > 0, "Expected TF-IDF term weights"
    print("Test 3 Passed: KNN Retriever classifies depth queries and explains TF-IDF terms.")
    
    # Test 4: Keyword Baseline
    baseline = KeywordBaselineMatcher(os.path.join(BASE_DIR, "data"))
    base_res = baseline.retrieve("Is the water quality safe for drinking?")
    assert base_res["predicted_intent"] == "quality", f"Expected quality, got {base_res['predicted_intent']}"
    print("Test 4 Passed: KeywordBaselineMatcher classifies quality queries.")
    
    # Test 5: End-to-end Pipeline
    pipeline = GroundwaterRetrievalPipeline(os.path.join(BASE_DIR, "data"))
    pipe_res = pipeline.process_query("Show groundwater level trend over time for Edwards Aquifer site 293830098255501")
    assert pipe_res["intent"] == "trend", f"Expected trend, got {pipe_res['intent']}"
    assert pipe_res["target_site"]["site_no"] == "293830098255501"
    assert "timeseries_data" in pipe_res["structured_record"]
    assert pipe_res["top_k_lookups"][0]["lookup_key"] == "293830098255501::trend"
    assert len(pipe_res["decision_summary"]) > 20
    print("Test 5 Passed: End-to-end Retrieval Pipeline retrieves trend record & decision summary.")
    
    # Test 6: Evaluator & Benchmark
    evaluator = RetrievalEvaluator(os.path.join(BASE_DIR, "data"))
    bench = evaluator.run_benchmark()
    assert bench["knn"]["intent_accuracy"] >= 0.85, f"Expected >= 85% accuracy, got {bench['knn']['intent_accuracy']}"
    assert bench["knn"]["exact_record_retrieval"]["mean_p_at_1"] >= 0.85
    assert bench["knn"]["category_retrieval"]["mean_p_at_1"] >= bench["keyword_baseline"]["category_retrieval"]["mean_p_at_1"]
    print(f"Test 6 Passed: Evaluator benchmark (KNN Acc: {bench['knn']['intent_accuracy'] * 100:.1f}%, Record P@1: {bench['knn']['exact_record_retrieval']['mean_p_at_1']:.4f}).")

    # Test 7: USGS NWIS Provenance & RDB Parser
    prov = get_dataset_provenance(os.path.join(BASE_DIR, "data"))
    assert prov["academic_id"] == "OPEN-DATA-USGS"
    assert prov["restricted_data_required"] is False
    sample_rdb = "# comment\nsite_no\tstation_nm\n15s\t50s\n01646500\tTest Well\n"
    parsed = _parse_usgs_rdb(sample_rdb)
    assert len(parsed) == 1 and parsed.iloc[0]["site_no"] == "01646500"
    print("Test 7 Passed: USGS NWIS provenance & RDB parser verified.")
    
    print("\nALL 7 TEST SUITES PASSED CLEANLY!")

if __name__ == "__main__":
    run_tests()
