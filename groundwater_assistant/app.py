import os
import sys
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval_pipeline import GroundwaterRetrievalPipeline
from src.evaluator import RetrievalEvaluator
from src.usgs_nwis_fetcher import get_dataset_provenance, fetch_nwis_sites

st.set_page_config(
    page_title="Groundwater Query Classification & Data Retrieval Assistant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.05rem;
        font-weight: 800;
        color: #0F2D59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.25rem;
    }
    .card-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.15rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .decision-banner {
        background: linear-gradient(90deg, #EFF6FF 0%, #F0FDF4 100%);
        border-left: 5px solid #0F2D59;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #0F172A;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-safe { background-color: #DCFCE7; color: #166534; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
    .badge-danger { background-color: #FEE2E2; color: #991B1B; }
    .badge-info { background-color: #E0F2FE; color: #0369A1; }
    .stat-title { font-size: 0.82rem; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .stat-value { font-size: 1.55rem; font-weight: 700; color: #0F172A; }
    .stat-sub { font-size: 0.8rem; color: #64748B; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return GroundwaterRetrievalPipeline(data_dir=os.path.join(BASE_DIR, "data"))


@st.cache_resource
def load_evaluator():
    return RetrievalEvaluator(data_dir=os.path.join(BASE_DIR, "data"))


pipeline = load_pipeline()
evaluator = load_evaluator()

# Offline-safe SVG/HTML badge in sidebar
st.sidebar.markdown("""
<div style="background:#0F2D59;color:#FFFFFF;padding:12px 14px;border-radius:10px;margin-bottom:12px;">
  <div style="font-size:0.75rem;letter-spacing:0.08em;color:#93C5FD;font-weight:700;">OPEN-DATA-USGS</div>
  <div style="font-size:1.1rem;font-weight:800;margin-top:2px;">💧 Groundwater Assistant</div>
  <div style="font-size:0.78rem;color:#CBD5E1;margin-top:4px;">CO2 • CO3 • CO5 | TF-IDF + Cosine KNN</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard Mode:",
    [
        "🔍 Natural Language Query Assistant",
        "🌍 Location Hydro-Soil Dossier (All-in-One)",
        "📊 Benchmark & Evaluation (P@k)",
        "🗄️ USGS NWIS Dataset Explorer",
        "ℹ️ System Architecture & Rubric",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("""
**Structured Categories**:
- 📍 `location`: Coordinates, county, state, aquifer
- 📏 `depth`: Water table depth (ft) & NAVD88 elev.
- 📈 `trend`: 24-month hydrograph & net change
- 🧪 `quality`: pH, nitrate, TDS, EPA potability

**Open Dataset**:
[USGS NWIS Portal](https://waterdata.usgs.gov/nwis)
""")


def render_structured_record(site: dict, intent: str, rec: dict, dossier: dict):
    """Renders the matched USGS groundwater record for the predicted intent."""
    st.subheader(f"Retrieved Structured USGS Record — [{intent.upper()}]")
    st.caption(
        f"USGS Site `{site['site_no']}` • **{site['station_nm']}** • "
        f"{site['county']} County, {site['state']} • Aquifer: **{site['aquifer']}**"
    )

    if intent == "depth":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Water Depth", f"{rec['current_water_level_ft']} ft")
        c2.metric("Water Level Elevation", f"{rec['water_level_elevation_ft']} ft NAVD88")
        c3.metric("Land Surface Altitude", f"{rec['altitude_ft']} ft")
        c4.metric("Surface Soil Moisture", f"{dossier['surface_soil_moisture_pct']}%")
        st.success(
            f"Water table is at **{rec['current_water_level_ft']} ft** below land surface at "
            f"{site['station_nm']} ({site['county']} County, {site['state']}). Status: {rec['measurement_status']}."
        )

    elif intent == "trend":
        ts = rec.get("timeseries_data")
        if ts is not None and not ts.empty:
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("24-Month Net Change", f"{rec['net_change_ft']:+.2f} ft")
            t2.metric("Trend Direction", rec["trend_direction"].split("(")[0].strip())
            t3.metric("Period Start", rec["start_date"])
            t4.metric("Period End", rec["end_date"])
            st.line_chart(ts.set_index("record_date")[["water_level_depth_ft"]].rename(
                columns={"water_level_depth_ft": "Depth to Water (ft below surface)"}
            ))
        else:
            st.info("No historical time-series records found for this station.")

    elif intent == "quality":
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("pH Level", f"{rec.get('ph', dossier['ph'])}", "Normal: 6.5–8.5")
        q2.metric("Nitrate (NO3)", f"{rec.get('nitrate_mg_l', dossier['nitrate_mg_l'])} mg/L", "EPA Limit: 10 mg/L")
        q3.metric("TDS", f"{rec.get('total_dissolved_solids_mg_l', dossier['total_dissolved_solids_mg_l'])} mg/L")
        q4.metric("Conductance", f"{rec.get('specific_conductance_us_cm', dossier['specific_conductance_us_cm'])} µS/cm")
        potability = rec.get("epa_potability", dossier["epa_potability"])
        q5.metric("EPA Potability", potability)
        assessment = rec.get("water_quality_assessment", dossier["water_quality_assessment"])
        if potability == "Compliant":
            st.success(f"Quality Assessment: **{assessment}** (Compliant with EPA primary & secondary standards).")
        else:
            st.warning(f"Quality Assessment: **{assessment}** ({potability}).")

    elif intent == "location":
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("County / State", f"{site['county']}, {site['state']}")
        l2.metric("Latitude / Longitude", f"{rec['latitude']}, {rec['longitude']}")
        l3.metric("Well Depth", f"{rec['well_depth_ft']} ft")
        l4.metric("Surface Altitude", f"{rec['altitude_ft']} ft")
        st.info(f"Aquifer System: **{site['aquifer']}** | Station Status: **{rec['status']}**")
        st.map(pd.DataFrame([{"lat": rec["latitude"], "lon": rec["longitude"]}]), zoom=6)


# ================================================================
# VIEW 1 (DEFAULT): NATURAL LANGUAGE QUERY ASSISTANT (TF-IDF + KNN)
# ================================================================
if page == "🔍 Natural Language Query Assistant":
    st.markdown('<div class="main-header">🔍 Groundwater Query Classification & Retrieval Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Type any plain-language groundwater question. '
        'The TF-IDF + Cosine KNN pipeline normalizes your query, classifies its retrieval category, '
        'and returns the matching USGS NWIS groundwater record directly.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Try Sample Field-Worker Questions:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    if "nl_query" not in st.session_state:
        st.session_state["nl_query"] = "How far down do we have to drill to reach groundwater at 342500118250001?"
    if qc1.button("📍 Location (Station 01646500)"):
        st.session_state["nl_query"] = "Where is groundwater monitoring station 01646500 situated?"
    if qc2.button("📏 Depth (Kern County Well)"):
        st.session_state["nl_query"] = "How far down do we have to drill to reach groundwater at 342500118250001?"
    if qc3.button("📈 Trend (Edwards Aquifer)"):
        st.session_state["nl_query"] = "Show me seasonal drawdown time-series for Edwards Aquifer station 293830098255501"
    if qc4.button("🧪 Quality (Fresno / Ogallala)"):
        st.session_state["nl_query"] = "Check pH, salinity, and nitrate levels for well 405012100123401"

    user_query = st.text_input("Enter plain-language groundwater question:", key="nl_query")
    k_val = st.slider("Number of KNN neighbors (k):", 1, 10, 5)

    if user_query:
        with st.spinner("Normalizing query and running TF-IDF + Cosine KNN retrieval..."):
            out = pipeline.process_query(user_query, k=k_val)
            intent = out["intent"]
            confidence = out["confidence"]
            site = out["target_site"]
            rec = out["structured_record"]
            dossier = out["location_dossier"]

        # Field-worker Decision Loop Banner
        st.markdown(
            f'<div class="decision-banner"><strong>✅ Field Decision Summary:</strong> {out["decision_summary"]}</div>',
            unsafe_allow_html=True,
        )

        # Direct Matching Structured Record
        render_structured_record(site, intent, rec, dossier)

        st.divider()

        # NLP Normalization, Classification & Baseline Comparison
        st.subheader("Pipeline Breakdown: Normalization, TF-IDF & KNN vs Keyword Baseline")
        col_a, col_b, col_c = st.columns([1.4, 1.3, 1.3])
        with col_a:
            st.markdown("**1. Query Normalization (CO2)**")
            st.markdown(f"- **Normalized Tokens**: `{out['normalized_query']}`")
            st.markdown(f"- **Intent Tokens (Entity-Disentangled)**: `{out['intent_normalized_query']}`")
            extracted = {k: v for k, v in out["entities"].items() if v is not None}
            st.markdown("**Extracted Structured Entities**:")
            st.json(extracted if extracted else {"note": "No explicit site/place entity; resolved via top KNN match"})

        with col_b:
            st.markdown("**2. TF-IDF + Cosine KNN Classification (CO3)**")
            st.markdown(f"- **Predicted Category**: `{intent.upper()}`")
            st.markdown(f"- **KNN Confidence**: `{confidence * 100:.1f}%`")
            st.progress(float(confidence))
            prob_df = pd.DataFrame(
                [{"Category": c.capitalize(), "Probability": p} for c, p in out["intent_probabilities"].items()]
            ).set_index("Category")
            st.bar_chart(prob_df)

        with col_c:
            st.markdown("**3. Keyword Baseline & Target Lookup**")
            st.markdown(f"- **Keyword Baseline**: `{out['baseline_intent'].upper()}` ({out['baseline_confidence'] * 100:.1f}%)")
            agreement = "Models Agree" if out["baseline_intent"] == intent else "KNN Disambiguated Beyond Keywords"
            st.info(f"**Comparison**: {agreement}\n\n**Target Station**: `{site['site_no']}` ({site['station_nm']})")

        with st.expander(f"Inspect Top {k_val} Nearest Training Neighbors (Cosine Similarity) & TF-IDF Weights"):
            tcol1, tcol2 = st.columns([1.8, 1.0])
            with tcol1:
                st.markdown("**Top-k Nearest Neighbors in TF-IDF Cosine Space**")
                st.dataframe(pd.DataFrame(out["top_k_neighbors"]))
            with tcol2:
                st.markdown("**Query TF-IDF Feature Weights**")
                if out["tfidf_terms"]:
                    st.dataframe(pd.DataFrame(out["tfidf_terms"]))
                else:
                    st.caption("No in-vocabulary TF-IDF terms matched.")

        with st.expander("Inspect Ranked Structured-Data Lookups & Complete Station Dossier"):
            st.markdown("**Top-k Ranked `(site_no, category)` Structured Lookups**")
            st.dataframe(pd.DataFrame(out["top_k_lookups"]))
            st.json({
                "site_no": dossier["site_no"],
                "station_nm": dossier["station_nm"],
                "current_water_level_ft": dossier["current_water_level_ft"],
                "net_change_ft": dossier["net_change_ft"],
                "trend_direction": dossier["trend_direction"],
                "ph": dossier["ph"],
                "nitrate_mg_l": dossier["nitrate_mg_l"],
                "epa_potability": dossier["epa_potability"],
            })

# ================================================================
# VIEW 2: ALL-IN-ONE LOCATION HYDRO-SOIL DOSSIER
# ================================================================
elif page == "🌍 Location Hydro-Soil Dossier (All-in-One)":
    st.markdown('<div class="main-header">🌍 Complete Place Groundwater & Soil Moisture Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select or type any place/region to inspect groundwater depth, soil moisture, 24-month trend, and drinking water quality together.</div>', unsafe_allow_html=True)

    st.markdown("**Quick Place Selector (India & Global):**")
    st.markdown("*🇮🇳 Indian States & Districts:*")
    pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
    if "dossier_place" not in st.session_state:
        st.session_state["dossier_place"] = "Punjab Ludhiana"
    if pcol1.button("🌾 Punjab (Ludhiana)"):
        st.session_state["dossier_place"] = "Punjab Ludhiana"
    if pcol2.button("🏜️ Rajasthan (Jodhpur)"):
        st.session_state["dossier_place"] = "Jodhpur Rajasthan"
    if pcol3.button("⛰️ Maharashtra (Pune)"):
        st.session_state["dossier_place"] = "Pune Maharashtra"
    if pcol4.button("🌊 UP (Varanasi Ganga)"):
        st.session_state["dossier_place"] = "Varanasi Uttar Pradesh"
    if pcol5.button("🏛️ Delhi NCR"):
        st.session_state["dossier_place"] = "New Delhi"
    if pcol6.button("💻 Bengaluru (Karnataka)"):
        st.session_state["dossier_place"] = "Bengaluru Karnataka"

    st.markdown("*🌴 Other Regions & Global:*")
    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    if gcol1.button("🥥 Kerala (Ernakulam)"):
        st.session_state["dossier_place"] = "Kochi Ernakulam Kerala"
    if gcol2.button("🌾 West Bengal (Kolkata)"):
        st.session_state["dossier_place"] = "Kolkata West Bengal"
    if gcol3.button("📍 California (Kern CA)"):
        st.session_state["dossier_place"] = "Kern County California"
    if gcol4.button("📍 Texas (Edwards TX)"):
        st.session_state["dossier_place"] = "Bexar County Texas Edwards Aquifer"
    if gcol5.button("📍 Florida (Marion FL)"):
        st.session_state["dossier_place"] = "Marion County Florida"

    place_input = st.text_input(
        "Enter name of place, county, state, or aquifer:",
        key="dossier_place",
        help="Type any region name e.g. 'California', 'Texas', 'Fresno', 'San Antonio', 'Phoenix', 'Florida', 'Nebraska', etc.",
    )

    if place_input:
        with st.spinner(f"Retrieving comprehensive groundwater, soil, and quality dossier for '{place_input}'..."):
            res = pipeline.process_query(place_input)
            dossier = res["location_dossier"]
            site = res["target_site"]

        st.markdown(f"### 📋 Hydrogeological & Soil Dossier: **{site['station_nm']}**")
        st.caption(f"📍 Location: **{site['county']} County, {site['state']}** | Aquifer: **{site['aquifer']}** | USGS Site ID: `{site['site_no']}`")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="card-box">
                <div class="stat-title">💧 Groundwater Level Depth</div>
                <div class="stat-value" style="font-size:1.45rem;">{dossier['current_water_level_ft']} ft <span style="font-size:1.05rem;color:#475569;">({dossier.get('current_water_level_m', round(dossier['current_water_level_ft']*0.3048, 1))} m)</span></div>
                <div class="stat-sub">Depth below ground surface</div>
                <div style="margin-top:8px;">
                    <span class="badge-pill badge-info">Elevation: {dossier['water_level_elevation_ft']} ft NAVD88</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            sm_val = dossier["surface_soil_moisture_pct"]
            sm_badge = "badge-danger" if sm_val < 15 else "badge-warning" if sm_val < 25 else "badge-safe"
            st.markdown(f"""
            <div class="card-box">
                <div class="stat-title">🌱 Surface Soil Moisture (0-10cm)</div>
                <div class="stat-value">{sm_val}%</div>
                <div class="stat-sub">Volumetric topsoil water content</div>
                <div style="margin-top:8px;">
                    <span class="badge-pill {sm_badge}">{dossier['soil_moisture_condition']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            rz_val = dossier["rootzone_soil_moisture_pct"]
            st.markdown(f"""
            <div class="card-box">
                <div class="stat-title">🌾 Root-Zone Moisture (10-100cm)</div>
                <div class="stat-value">{rz_val}%</div>
                <div class="stat-sub">Agricultural root availability</div>
                <div style="margin-top:8px;">
                    <span class="badge-pill badge-info">Soil: {dossier['soil_type']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            scarcity = dossier["water_scarcity_index"]
            sc_badge = "badge-danger" if "Overdraft" in scarcity or "High" in scarcity else "badge-warning" if "Moderate" in scarcity else "badge-safe"
            st.markdown(f"""
            <div class="card-box">
                <div class="stat-title">⚠️ Water Scarcity / Risk Index</div>
                <div class="stat-value" style="font-size:1.25rem;">{scarcity.split('/')[0]}</div>
                <div class="stat-sub">Aquifer drawdown risk assessment</div>
                <div style="margin-top:8px;">
                    <span class="badge-pill {sc_badge}">{dossier['trend_direction']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        r2_col1, r2_col2 = st.columns([1.8, 1.2])
        with r2_col1:
            st.markdown("#### 📈 24-Month Groundwater Fluctuation & Hydrograph Trend")
            ts = dossier.get("timeseries_data")
            if ts is not None and not ts.empty:
                chart_df = ts.set_index("record_date")[["water_level_depth_ft"]].rename(
                    columns={"water_level_depth_ft": "Depth to Water (ft)"}
                )
                st.line_chart(chart_df)
                st.info(
                    f"**Trend Summary**: Over the last 24 months, the water table shifted by **{dossier['net_change_ft']:+.2f} ft** "
                    f"(Min: {dossier['min_depth_ft']} ft, Max: {dossier['max_depth_ft']} ft). Status: **{dossier['trend_direction']}**."
                )
            else:
                st.info("No time series data available for this station.")

        with r2_col2:
            st.markdown("#### 🌱 Soil Moisture & Drought Diagnostic")
            st.write(f"**Topsoil Moisture (0-10 cm)**: `{dossier['surface_soil_moisture_pct']}%`")
            st.progress(float(min(1.0, dossier["surface_soil_moisture_pct"] / 50.0)))
            st.write(f"**Root-Zone Moisture (10-100 cm)**: `{dossier['rootzone_soil_moisture_pct']}%`")
            st.progress(float(min(1.0, dossier["rootzone_soil_moisture_pct"] / 50.0)))
            st.markdown(f"""
            - **Soil Classification**: `{dossier['soil_type']}`
            - **Condition Rating**: `{dossier['soil_moisture_condition']}`
            - **Total Well Drilling Depth**: `{dossier['well_depth_ft']} ft`
            - **Surface Altitude**: `{dossier['altitude_ft']} ft NAVD88`
            """)

        st.markdown("#### 🧪 Groundwater Chemical Quality & Drinking Potability")
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("pH Level", f"{dossier['ph']}", "Normal: 6.5-8.5")
        q2.metric("Nitrate (NO3)", f"{dossier['nitrate_mg_l']} mg/L", delta="EPA Limit: 10 mg/L", delta_color="inverse" if dossier["nitrate_mg_l"] > 10 else "normal")
        q3.metric("Total Dissolved Solids", f"{dossier['total_dissolved_solids_mg_l']} mg/L", "EPA Secondary: 500 mg/L")
        q4.metric("Specific Conductance", f"{dossier['specific_conductance_us_cm']} µS/cm")
        potability_badge = "badge-safe" if dossier["epa_potability"] == "Compliant" else "badge-danger"
        with q5:
            st.markdown("**EPA Potability**")
            st.markdown(f'<span class="badge-pill {potability_badge}">{dossier["epa_potability"]}</span>', unsafe_allow_html=True)
            st.caption(dossier["water_quality_assessment"])

        st.markdown("#### 📍 Monitoring Station Geographic Position")
        st.map(pd.DataFrame([{"lat": dossier["latitude"], "lon": dossier["longitude"]}]), zoom=7)

# ================================================================
# VIEW 3: BENCHMARK & EVALUATION (Precision@k + Manual Checks)
# ================================================================
elif page == "📊 Benchmark & Evaluation (P@k)":
    st.markdown('<div class="main-header">📊 Model Benchmark & Ranking-Quality Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Validating TF-IDF + Cosine KNN against the Keyword-Matching Baseline '
        'using Precision@k, MRR, and manual relevance checks across semantically identical paraphrases (CO5).</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Computing evaluation metrics..."):
        bench = evaluator.run_benchmark()

    knn = bench["knn"]
    base = bench["keyword_baseline"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Intent Accuracy (KNN)", f"{knn['intent_accuracy'] * 100:.1f}%", delta=f"{(knn['intent_accuracy'] - base['intent_accuracy']) * 100:+.1f}% vs Baseline")
    m2.metric("Category P@1 (KNN)", f"{knn['category_retrieval']['mean_p_at_1']:.4f}", delta=f"{knn['category_retrieval']['mean_p_at_1'] - base['category_retrieval']['mean_p_at_1']:+.4f}")
    m3.metric("Category P@3 (KNN)", f"{knn['category_retrieval']['mean_p_at_3']:.4f}", delta=f"{knn['category_retrieval']['mean_p_at_3'] - base['category_retrieval']['mean_p_at_3']:+.4f}")
    m4.metric("Category P@5 (KNN)", f"{knn['category_retrieval']['mean_p_at_5']:.4f}", delta=f"{knn['category_retrieval']['mean_p_at_5'] - base['category_retrieval']['mean_p_at_5']:+.4f}")
    m5.metric("Exact Record P@1", f"{knn['exact_record_retrieval']['mean_p_at_1']:.4f}", delta=f"{knn['exact_record_retrieval']['mean_p_at_1'] - base['exact_record_retrieval']['mean_p_at_1']:+.4f}")

    st.markdown("---")
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Category Neighbor Ranking Precision@k & MRR")
        cat_chart = pd.DataFrame({
            "Metric": ["P@1", "P@3", "P@5", "MRR"],
            "TF-IDF + KNN": [
                knn["category_retrieval"]["mean_p_at_1"],
                knn["category_retrieval"]["mean_p_at_3"],
                knn["category_retrieval"]["mean_p_at_5"],
                knn["category_retrieval"]["mrr"],
            ],
            "Keyword Baseline": [
                base["category_retrieval"]["mean_p_at_1"],
                base["category_retrieval"]["mean_p_at_3"],
                base["category_retrieval"]["mean_p_at_5"],
                base["category_retrieval"]["mrr"],
            ],
        }).set_index("Metric")
        st.bar_chart(cat_chart)

    with c_right:
        st.subheader("Structured Lookup (Site + Category) P@1 & MRR")
        rec_chart = pd.DataFrame({
            "Metric": ["Record P@1", "Record MRR", "Intent Accuracy"],
            "TF-IDF + KNN": [
                knn["exact_record_retrieval"]["mean_p_at_1"],
                knn["exact_record_retrieval"]["mrr"],
                knn["intent_accuracy"],
            ],
            "Keyword Baseline": [
                base["exact_record_retrieval"]["mean_p_at_1"],
                base["exact_record_retrieval"]["mrr"],
                base["intent_accuracy"],
            ],
        }).set_index("Metric")
        st.bar_chart(rec_chart)

    st.subheader("Manual Relevance Checks Across Semantically Identical Paraphrase Groups")
    st.caption(f"Evaluated {bench['total_test_queries']} unseen paraphrased queries across {len(bench['group_summaries'])} test groups.")
    for g in bench["group_summaries"]:
        pass_pct = g.get("manual_check_pass_rate", 0.0) * 100
        with st.expander(f"📁 {g['group_name']} [Expected: {g['category'].upper()}, Target: {g['target_site']}] — Pass Rate: {pass_pct:.0f}%"):
            st.dataframe(pd.DataFrame(g["queries_evaluated"]))

# ================================================================
# VIEW 4: USGS NWIS DATASET EXPLORER
# ================================================================
elif page == "🗄️ USGS NWIS Dataset Explorer":
    st.markdown('<div class="main-header">🗄️ USGS NWIS Open Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect the publicly available USGS National Water Information System (`OPEN-DATA-USGS`) snapshots and live API connector.</div>', unsafe_allow_html=True)

    prov = get_dataset_provenance(os.path.join(BASE_DIR, "data"))
    p1, p2, p3 = st.columns([1.4, 1.6, 1.0])
    p1.info(f"**Academic ID**: `{prov['academic_id']}`\n\n**Portal**: {prov['portal_url']}")
    p2.info(f"**Site API**: `{prov['site_api']}`\n\n**GW Levels API**: `{prov['gwlevels_api']}`")
    with p3:
        if st.button("🌐 Test Live USGS NWIS Fetch"):
            live_res = fetch_nwis_sites(["01646500"], timeout=5)
            if live_res["status"] == "ok":
                st.success(f"Connected to USGS NWIS live API ({len(live_res['dataframe'])} rows).")
            else:
                st.warning(f"Using local offline USGS snapshot ({live_res.get('error', 'offline')}).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏛️ Monitoring Stations",
        "📈 24-Month Time Series",
        "🧪 Water Quality Profiles",
        f"💬 Training Queries ({len(pipeline.retriever.train_df)})",
    ])
    with tab1:
        st.dataframe(pipeline.data_loader.get_all_sites())
    with tab2:
        st.dataframe(pipeline.data_loader.timeseries_df)
    with tab3:
        st.dataframe(pipeline.data_loader.quality_df)
    with tab4:
        st.dataframe(pipeline.retriever.train_df[["query_text", "intent", "target_site_no", "intent_normalized_text"]])

# ================================================================
# VIEW 5: ARCHITECTURE & RUBRIC
# ================================================================
elif page == "ℹ️ System Architecture & Rubric":
    st.markdown('<div class="main-header">ℹ️ System Architecture & Course Outcomes</div>', unsafe_allow_html=True)
    st.markdown("""
### 📋 Project Specification Alignment (`OPEN-DATA-USGS`)
- **Dataset**: Publicly available USGS National Water Information System ([https://waterdata.usgs.gov/nwis](https://waterdata.usgs.gov/nwis)) — no restricted or confidential data required.
- **CO2 (Query Normalization & Vector Space)**: Lowercasing, stopword removal, entity-disentangled morphological normalization, and sublinear unigram+bigram TF-IDF representations aligned to four structured retrieval categories (`location`, `depth`, `trend`, `quality`).
- **CO3 (Cosine KNN Retrieval vs Keyword Baseline)**: Cosine-distance k-Nearest Neighbors classifier & retriever (`k=5`) paired with regex/place entity resolution, benchmarked against an exact lexical keyword-matching + Jaccard baseline.
- **CO5 (Ranking Evaluation & Field Decision Loop)**: Validated using `Precision@1`, `Precision@3`, `Precision@5`, `MRR`, and manual relevance checks across 8 semantically identical paraphrase groups (40 unseen queries), deployed in Streamlit so a field worker obtains the matching groundwater record directly without manual lookup.
""")
