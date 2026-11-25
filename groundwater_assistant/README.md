# Groundwater Query Classification and Data Retrieval Assistant System

**Academic Identifier**: OPEN-DATA-USGS  
**Course Outcomes**: CO2, CO3, CO5  
**Domain**: Hydrology / Natural Language Processing / Information Retrieval  
**Dataset Source**: [USGS National Water Information System (NWIS)](https://waterdata.usgs.gov/nwis)  

---

## 📌 Executive Summary
Field workers, hydrogeologists, environmental officers, and agricultural planners frequently need fast, authoritative access to groundwater measurements from the USGS National Water Information System. However, querying these systems traditionally requires memorizing cryptic alphanumeric station identifiers (e.g. `site_no = 342500118250001`), knowing specific parameter codes, or writing complex database queries.

This system bridges that gap:
1. **Normalizes** natural-language questions (handling colloquial variations, typos, and phrasing).
2. **Transforms** queries into TF-IDF unigram and bigram representations in a sublinear feature space.
3. **Classifies** queries into four structured data-retrieval categories using **k-Nearest Neighbors (KNN)** with **Cosine Similarity**:
   - `Location`: Spatial coordinates, state, county, surface elevation, well depth, and aquifer name.
   - `Depth`: Current water table depth below land surface (ft) and water level elevation (NAVD88).
   - `Trend`: Historical 24-month time-series hydrograph, drawdown/recharge rates, and net change.
   - `Quality`: Chemical parameters (pH, specific conductance, total dissolved solids, nitrate concentration) and EPA drinking water compliance.
4. **Benchmarks** the KNN retrieval pipeline against a strict **Keyword-Matching Baseline** using **Precision@k (P@1, P@3, P@5)**, **Mean Reciprocal Rank (MRR)**, and **paraphrase robustness tests**.
5. **Deploys** as an interactive, decision-support **Streamlit Web Dashboard**.

---

## 🎯 Course Outcomes (CO) Mapping

| Course Outcome | Implementation in this System |
|---|---|
| **CO2** (Data Preparation & Vector Representations) | Curated USGS NWIS structured datasets (`sites`, `timeseries`, `water_quality`); NLP normalization (lowercasing, stopword stripping, tokenization, regex entity extraction); Sublinear TF-IDF feature space (unigrams + bigrams). |
| **CO3** (Information Retrieval & Machine Learning) | Implementation of Cosine Distance KNN retriever (`k=5`); Category intent classifier; Parameter extraction (USGS Site ID, state, county, aquifer); Lexical keyword-matching baseline model. |
| **CO5** (Evaluation, Deployment & Decision Loop) | Ranking evaluation via Precision@1, Precision@3, Precision@5, and MRR; Semantic paraphrase testing across 6 test groups; Deployment of interactive Streamlit UI closing the field worker decision loop. |

---

## 📊 Benchmark Evaluation Results

Evaluated across **40 unseen, differently phrased, semantically identical test queries** spanning 8 domain groups (including both USGS site-ID queries and natural place-name queries):

| Metric | TF-IDF + Cosine KNN | Keyword Baseline | Relative Advantage |
|---|---|---|---|
| **Intent Classification Accuracy** | **100.0%** | 92.5% | **+7.5% higher accuracy** |
| **Category Precision @ 1 (P@1)** | **1.0000** | 0.9500 | **+5.0% higher top-1 precision** |
| **Category Precision @ 3 (P@3)** | **0.9500** | 0.8417 | **+10.8% higher top-3 precision** |
| **Category Precision @ 5 (P@5)** | **0.8750** | 0.7500 | **+12.5% higher top-5 precision** |
| **Category Mean Reciprocal Rank (MRR)** | **1.0000** | 0.9750 | **+2.5% higher MRR** |
| **Structured Record Lookup P@1** | **1.0000** | 0.9250 | **+7.5% higher exact lookup P@1** |
| **Structured Record Lookup MRR** | **1.0000** | 0.9625 | **+3.8% higher lookup MRR** |

> **Key Finding**: Disentangling numeric USGS station IDs and proper place names before TF-IDF vectorization prevents rare entity tokens from distorting cosine similarity, enabling the TF-IDF + Cosine KNN model to achieve 100% intent and rank-1 record retrieval across semantic paraphrases where strict keyword matching struggles.

---

## 🗂️ Project Directory Structure

```text
groundwater_assistant/
├── app.py                      # Interactive Streamlit Web Dashboard
├── evaluate.py                 # Standalone benchmark evaluation CLI script
├── generate_data.py            # USGS NWIS dataset & training/test query generator
├── test_system.py              # 7-suite unit & integration test runner
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & technical report
├── data/
│   ├── usgs_groundwater_sites.csv        # Monitoring stations metadata (lat, lon, aquifer, depth)
│   ├── usgs_water_levels_timeseries.csv  # 24-month water depth time-series
│   ├── usgs_water_quality.csv            # Lab water quality parameters & EPA potability
│   ├── training_queries.csv              # 128 balanced natural language queries across 4 categories
│   └── benchmark_test_queries.json       # 8 paraphrase groups (40 unseen queries) for P@k & MRR
└── src/
    ├── __init__.py
    ├── data_loader.py          # Structured USGS querying, record formatting & decision summaries
    ├── preprocessor.py         # Text normalization, entity disentanglement & regex extraction
    ├── knn_retriever.py        # TF-IDF vectorizer + Cosine KNN model & term explanation
    ├── keyword_baseline.py     # Lexical keyword matching + Jaccard baseline
    ├── evaluator.py            # Category & structured-record Precision@k, MRR, and manual checks
    ├── retrieval_pipeline.py   # End-to-end integration pipeline
    └── usgs_nwis_fetcher.py    # Public USGS NWIS REST API fetcher with offline cache fallback
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Evaluation
```bash
python evaluate.py
```
This generates the full terminal benchmark table and outputs `evaluation_report.json`.

### 3. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📱 Dashboard Capabilities
1. **Field Query Assistant**:
   - Accepts plain-language questions.
   - Normalizes text and extracts USGS site IDs, counties, aquifers, and states.
   - Shows predicted category with confidence score and class probability distribution.
   - Directly renders the structured groundwater answer:
     - 📍 **Location**: GPS coordinates, well depth, and interactive geographical map.
     - 📏 **Depth**: Water table depth below surface (ft) and elevation relative to NAVD88.
     - 📈 **Trend**: 24-month hydrograph line chart and recharge/depletion status.
     - 🧪 **Quality**: pH, nitrate, dissolved solids, conductivity, and EPA drinking water compliance.
2. **Benchmark & Evaluation Page**:
   - Real-time comparison between TF-IDF + KNN and the Keyword Baseline.
   - Precision@1, Precision@3, Precision@5 comparison charts.
   - Expandable semantic paraphrase test suites.
3. **Dataset Explorer**:
   - Interactive tables for USGS monitoring stations, time-series measurements, and water quality chemical profiles.
