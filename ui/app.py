import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="SafeMD.ai", page_icon="🏥", layout="wide")


def load_data(uploaded_file, fallback_path: Path) -> dict:
    if uploaded_file is not None:
        return json.load(uploaded_file)
    if fallback_path.exists():
        return json.loads(fallback_path.read_text(encoding="utf-8"))
    return {"facility_results": [], "dmaic_summary": {}, "equity_summary": {}}


def to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trust_score"] = pd.to_numeric(df.get("trust_score"), errors="coerce")
    df["missing_count"] = df.get("missing_fields", pd.Series(dtype=object)).apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    df["has_anomaly"] = df.get("anomaly_flags", pd.Series(dtype=object)).apply(
        lambda x: len(x) > 0 if isinstance(x, list) else False
    )
    if "facility_type_id" not in df.columns:
        df["facility_type_id"] = "UNKNOWN"
    df["facility_type_id"] = df["facility_type_id"].fillna("UNKNOWN").astype(str)
    return df


def classify_healthcare_types(row: dict) -> list[str]:
    text_blob = " ".join(
        [
            " ".join(row.get("confirmed_capabilities", []) or []),
            " ".join(row.get("unverified_claims", []) or []),
            str(row.get("reasoning_trace", "")),
        ]
    ).lower()
    mapping = {
        "Primary Care": ["primary", "family medicine", "outpatient", "general medicine"],
        "Emergency Services": ["emergency", "trauma", "critical care", "icu"],
        "Optometry/Ophthalmology": ["ophthalmology", "eye", "optometry", "retina", "glaucoma"],
        "Sports Medicine": ["sports", "orthopedic", "physiotherapy", "rehab"],
        "Dental": ["dental", "dentistry", "endodontics", "orthodontics"],
        "Oncology": ["oncology", "cancer", "chemotherapy"],
        "Dialysis/Nephrology": ["dialysis", "nephrology", "renal"],
    }
    found = [k for k, keywords in mapping.items() if any(w in text_blob for w in keywords)]
    return found if found else ["General Healthcare"]


def load_incidents(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "incident_id",
                "timestamp_utc",
                "facility_id",
                "facility_name",
                "incident_scope",
                "incident_type",
                "incident_subtypes",
                "description",
                "severity",
                "status",
            ]
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        if "incident_subtype" not in df.columns:
            df["incident_subtype"] = df.get("incident_subtypes", pd.Series(dtype=object)).apply(
                lambda x: ", ".join(x) if isinstance(x, list) else (x if isinstance(x, str) else "")
            )
    return df


def incident_summary_for_facility(incidents_df: pd.DataFrame, facility_id: int) -> tuple[list[str], int]:
    if incidents_df.empty:
        return [], 0
    subset = incidents_df[(incidents_df["facility_id"] == facility_id) & (incidents_df["status"] == "open")]
    subtypes = []
    for val in subset.get("incident_subtype", pd.Series(dtype=str)).tolist():
        if isinstance(val, str) and val.strip():
            subtypes.append(val.strip())
    for vals in subset.get("incident_subtypes", pd.Series(dtype=object)).tolist():
        if isinstance(vals, list):
            subtypes.extend([str(x).strip() for x in vals if str(x).strip()])
    return sorted(set(subtypes)), len(subset)


def infer_top_issue_clusters(incidents_df: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    if incidents_df.empty:
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "count"])
    text = (
        incidents_df["incident_type"].fillna("").astype(str)
        + " "
        + incidents_df["incident_subtype"].fillna("").astype(str)
        + " "
        + incidents_df["description"].fillna("").astype(str)
    ).str.lower()
    if text.str.strip().eq("").all():
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "count"])
    k = min(n_clusters, max(2, len(text)))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=800)
    X = vectorizer.fit_transform(text)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    terms = vectorizer.get_feature_names_out()
    centers = model.cluster_centers_

    rows = []
    for cid in range(k):
        count = int((labels == cid).sum())
        top_term_idx = centers[cid].argsort()[-3:][::-1]
        cluster_label = ", ".join([terms[i] for i in top_term_idx])
        rows.append({"cluster_id": cid, "cluster_label": cluster_label, "count": count})
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def write_incidents(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def seed_sample_incidents(df: pd.DataFrame, n: int = 50) -> list[dict]:
    if df.empty:
        return []
    random.seed(42)
    scopes = ["Single Patient", "Single Department", "Whole Hospital", "Regional Network", "Supply Chain", "Staff Safety"]
    types = [
        "Unexpected Outcome",
        "Insufficient Resources",
        "Insufficient Staff",
        "Equipment Failure",
        "Infrastructure Failure",
        "Near Miss",
    ]
    subtypes = [
        "Power outage",
        "Oxygen shortage",
        "No anesthesiologist",
        "Ventilator failure",
        "Nurse shortage",
        "Medication error",
        "Dialysis down",
        "Ambulance delay",
    ]
    severities = ["low", "medium", "high", "critical"]
    statuses = ["open", "monitoring", "resolved"]
    rows: list[dict] = []
    for i in range(n):
        r = df.sample(1, random_state=42 + i).iloc[0]
        rows.append(
            {
                "incident_id": f"inc_seed_{i+1}",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "facility_id": int(r["facility_id"]),
                "facility_name": r["facility_name"],
                "incident_scope": random.choice(scopes),
                "incident_type": random.choice(types),
                "incident_subtype": random.choice(subtypes),
                "description": "Seeded incident for local map and analysis testing.",
                "severity": random.choice(severities),
                "status": random.choices(statuses, weights=[0.6, 0.2, 0.2], k=1)[0],
            }
        )
    return rows


def dashboard_metrics(df: pd.DataFrame, incidents_df: pd.DataFrame):
    if df.empty:
        st.info("No facility data loaded.")
        return
    total = len(df)
    avg_trust = float(df["trust_score"].fillna(0).mean()) if "trust_score" in df.columns else 0.0
    urgent = int(df.get("urgent_need", pd.Series(dtype=bool)).fillna(False).sum())
    one_away = int(df.get("one_resource_away", pd.Series(dtype=bool)).fillna(False).sum())
    review = int(df.get("needs_human_review", pd.Series(dtype=bool)).fillna(False).sum())

    active_incidents = int((incidents_df["status"] == "open").sum()) if not incidents_df.empty else 0
    a, b, c, d, e, f = st.columns(6)
    a.metric("Facilities", f"{total:,}")
    b.metric("Avg Trust Score", f"{avg_trust:.1f}")
    c.metric("Urgent Need", urgent)
    d.metric("One Resource Away", one_away)
    e.metric("Needs Human Review", review)
    f.metric("Active Incidents", active_incidents)


def apply_filters(df: pd.DataFrame, incidents_df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Planning Filters")
    trust_min = st.sidebar.slider("Minimum Verification Confidence", 0, 100, 60)
    capability_gap_only = st.sidebar.checkbox("Show Capability Gap Risk Only", value=False)
    one_intervention_only = st.sidebar.checkbox("Show 'One Intervention Away' Facilities", value=False)
    audit_priority_only = st.sidebar.checkbox("Show Manual Audit Priority", value=False)
    state_filter = st.sidebar.multiselect(
        "State/Region",
        sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else [],
    )
    has_icu_gap = st.sidebar.checkbox("ICU Dependency Gap (no supporting infra evidence)", value=False)
    has_open_incident = st.sidebar.checkbox("Active Incident Facilities Only", value=False)
    all_service_types = sorted({s for row in df["healthcare_types"] for s in (row or [])})
    service_filter = st.sidebar.selectbox("Service Capability Category", ["All"] + all_service_types)

    st.sidebar.caption(
        "Capability Gap Risk = missing critical fields with reasonable confidence; "
        "One Intervention Away = likely one missing resource to become functional; "
        "Manual Audit Priority = validator flagged for human review."
    )

    filtered = df[df["trust_score"].fillna(0) >= trust_min]
    if capability_gap_only and "urgent_need" in filtered.columns:
        filtered = filtered[filtered["urgent_need"] == True]  # noqa: E712
    if one_intervention_only and "one_resource_away" in filtered.columns:
        filtered = filtered[filtered["one_resource_away"] == True]  # noqa: E712
    if audit_priority_only and "needs_human_review" in filtered.columns:
        filtered = filtered[filtered["needs_human_review"] == True]  # noqa: E712
    if state_filter and "state" in filtered.columns:
        filtered = filtered[filtered["state"].isin(state_filter)]
    if has_icu_gap:
        filtered = filtered[
            filtered["anomaly_flags"].apply(
                lambda x: "ICU_WITHOUT_SUPPORTING_INFRA" in x if isinstance(x, list) else False
            )
        ]
    if has_open_incident and not incidents_df.empty:
        ids = incidents_df[incidents_df["status"] == "open"]["facility_id"].unique().tolist()
        filtered = filtered[filtered["facility_id"].isin(ids)]
    if service_filter != "All":
        filtered = filtered[
            filtered["healthcare_types"].apply(
                lambda x: service_filter in x if isinstance(x, list) else False
            )
        ]
    return filtered


def render_map(filtered_df: pd.DataFrame, incidents_df: pd.DataFrame):
    st.subheader("Live Ground Truth Facility Map")
    map_df = filtered_df.copy()
    if map_df.empty:
        st.info("No records match current filters.")
        return
    coords = pd.json_normalize(map_df["coordinates"]).rename(columns={"lat": "lat", "lon": "lon"})
    map_df = pd.concat([map_df.reset_index(drop=True), coords], axis=1)
    map_df = map_df.dropna(subset=["lat", "lon"])
    if map_df.empty:
        st.info("No geocoded coordinates in current selection.")
        return
    map_df["active_incident_subtypes"], map_df["active_incident_count"] = zip(
        *map_df["facility_id"].apply(lambda x: incident_summary_for_facility(incidents_df, x))
    )
    map_df["has_power_outage"] = map_df["active_incident_subtypes"].apply(
        lambda x: any("power outage" in str(v).lower() for v in x) if isinstance(x, list) else False
    )
    map_df["marker_color"] = map_df.apply(
        lambda r: [255, 0, 0]
        if r["has_power_outage"]
        else ([255, 140, 0] if r["active_incident_count"] > 0 else [0, 160, 80]),
        axis=1,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        # Pixel-based radii prevent oversized circles at high zoom.
        get_radius=8,
        radius_units="pixels",
        radius_min_pixels=2,
        radius_max_pixels=14,
        get_fill_color="marker_color",
        pickable=True,
    )
    tooltip = {
        "html": "<b>{facility_name}</b><br/>Trust: {trust_score}<br/>Open Incidents: {active_incident_count}<br/>Types: {active_incident_subtypes}",
        "style": {"backgroundColor": "white", "color": "black"},
    }
    view_state = pdk.ViewState(
        latitude=float(map_df["lat"].mean()),
        longitude=float(map_df["lon"].mean()),
        zoom=4.5,
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    st.caption("Marker colors: red = power outage, orange = other open incidents, green = no active incidents")


def render_dmaic(dmaic: dict):
    st.subheader("Executive Dashboard (DMAIC / A3)")
    if not dmaic:
        st.info("No DMAIC summary loaded.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Define**")
        st.write(dmaic.get("define", "N/A"))
        st.markdown("**Analyze**")
        for x in dmaic.get("analyze", []):
            st.write(f"- {x}")
        st.markdown("**Improve**")
        for x in dmaic.get("improve", []):
            st.write(f"- {x}")
    with c2:
        st.markdown("**Measure**")
        measures = dmaic.get("measure", [])
        if measures:
            st.dataframe(pd.DataFrame(measures), use_container_width=True)
        st.markdown("**Control**")
        for x in dmaic.get("control", []):
            st.write(f"- {x}")
        st.markdown("**District Risk Signals**")
        risks = dmaic.get("district_risk_signals", [])
        if risks:
            st.dataframe(pd.DataFrame(risks), use_container_width=True)


def render_facility_detail(filtered_df: pd.DataFrame):
    st.subheader("Facility Intelligence + Source Traceability")
    if filtered_df.empty:
        st.info("No facility selected.")
        return
    options = (
        filtered_df["facility_name"].fillna("UNKNOWN")
        + " | "
        + filtered_df["city"].fillna("UNKNOWN")
        + ", "
        + filtered_df["state"].fillna("UNKNOWN")
    ).tolist()
    selected_label = st.selectbox("Select Facility", options)
    idx = options.index(selected_label)
    row = filtered_df.iloc[idx].to_dict()

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"### {row.get('facility_name', 'UNKNOWN')}")
        st.write(f"**Location:** {row.get('city', 'UNKNOWN')}, {row.get('state', 'UNKNOWN')} ({row.get('pincode', 'UNKNOWN')})")
        st.write(f"**Trust Score:** {row.get('trust_score', 'N/A')}")
        st.write(f"**Reasoning Trace:** {row.get('reasoning_trace', '')}")
        st.markdown("**Confirmed Capabilities**")
        for x in row.get("confirmed_capabilities", []):
            st.write(f"- {x}")
        st.markdown("**Unverified Claims**")
        for x in row.get("unverified_claims", []):
            st.write(f"- {x}")
        st.markdown("**Missing Fields**")
        for x in row.get("missing_fields", []):
            st.write(f"- {x}")
    with right:
        st.markdown("**Flags**")
        st.write(f"Urgent Need: {row.get('urgent_need', False)}")
        st.write(f"One Resource Away: {row.get('one_resource_away', False)}")
        st.write(f"Needs Review: {row.get('needs_human_review', False)}")
        for flag in row.get("anomaly_flags", []):
            st.write(f"- {flag}")

    st.markdown("#### Evidence Snippets (Source Citation)")
    evidence = row.get("evidence", [])
    if evidence:
        st.dataframe(pd.DataFrame(evidence), use_container_width=True)
    else:
        st.info("No evidence snippets available.")

    with st.expander("Correct This Entry (Feedback Loop)"):
        correction_text = st.text_area("What should be corrected?")
        corrected_field = st.text_input("Field to correct (optional)")
        submit = st.button("Submit Correction")
        if submit and correction_text.strip():
            feedback_path = Path("ui/feedback_submissions.jsonl")
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "facility_id": row.get("facility_id"),
                "facility_name": row.get("facility_name"),
                "field": corrected_field.strip() or None,
                "correction_text": correction_text.strip(),
            }
            with feedback_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            st.success("Correction saved locally. Wire this to your backend ingestion endpoint next.")


def render_incident_reporting_tab(df: pd.DataFrame, incidents_path: Path):
    st.subheader("Incident Reporting")
    if df.empty:
        st.info("Load facility data first.")
        return
    options = (
        df["facility_name"].fillna("UNKNOWN") + " | " + df["city"].fillna("UNKNOWN") + ", " + df["state"].fillna("UNKNOWN")
    ).tolist()
    selected_label = st.selectbox("Facility", options, key="incident_facility")
    idx = options.index(selected_label)
    facility = df.iloc[idx].to_dict()

    incident_scope = st.radio(
        "Incident Scope",
        ["Single Patient", "Single Department", "Whole Hospital", "Regional Network", "Supply Chain", "Staff Safety"],
        horizontal=True,
    )
    incident_type = st.selectbox(
        "Primary Incident Type",
        ["Unexpected Outcome", "Insufficient Resources", "Insufficient Staff", "Equipment Failure", "Infrastructure Failure", "Near Miss"],
    )
    incident_subtype = st.text_input(
        "Incident Subtype (2-3 words)",
        placeholder="e.g. Power outage, Oxygen leak, Staff shortage",
    )
    severity = st.select_slider("Severity", options=["low", "medium", "high", "critical"], value="medium")
    description = st.text_area("Incident Description", height=180)
    status = st.selectbox("Status", ["open", "monitoring", "resolved"], index=0)

    if st.button("Submit Incident Report", type="primary"):
        if not description.strip():
            st.error("Please enter an incident description.")
            return
        payload = {
            "incident_id": f"inc_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "facility_id": int(facility["facility_id"]),
            "facility_name": facility["facility_name"],
            "incident_scope": incident_scope,
            "incident_type": incident_type,
            "incident_subtype": incident_subtype.strip(),
            "description": description.strip(),
            "severity": severity,
            "status": status,
        }
        incidents_path.parent.mkdir(parents=True, exist_ok=True)
        with incidents_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        st.success("Incident submitted. Live map and analysis update on refresh.")

    st.markdown("#### Incident Dataset Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear All Incident Reports"):
            write_incidents(incidents_path, [])
            st.success("All incidents cleared.")
    with c2:
        if st.button("Seed Exactly 50 Incidents"):
            rows = seed_sample_incidents(df, n=50)
            write_incidents(incidents_path, rows)
            st.success("Replaced incident dataset with exactly 50 seeded incidents.")


def render_incident_analysis_tab(df: pd.DataFrame, incidents_df: pd.DataFrame):
    st.subheader("Incident Analysis")
    if df.empty:
        st.info("No facilities loaded.")
        return
    if incidents_df.empty:
        st.info("No incidents reported yet.")
        return

    facility_options = (
        df["facility_name"].fillna("UNKNOWN") + " | " + df["city"].fillna("UNKNOWN") + ", " + df["state"].fillna("UNKNOWN")
    ).tolist()
    selected_label = st.selectbox("Choose Hospital", facility_options, key="analysis_facility")
    idx = facility_options.index(selected_label)
    facility_id = int(df.iloc[idx]["facility_id"])
    facility_name = df.iloc[idx]["facility_name"]

    subset = incidents_df[incidents_df["facility_id"] == facility_id].copy()
    st.markdown(f"### {facility_name}")
    if subset.empty:
        st.info("No incidents for this hospital.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidents", len(subset))
    col2.metric("Open Incidents", int((subset["status"] == "open").sum()))
    col3.metric("Critical Incidents", int((subset["severity"] == "critical").sum()))

    st.markdown("**Overview of Trends + Suggested Actionables**")
    # Placeholder for future agent-generated analysis:
    # trends_summary = call_agent_for_incident_summary(subset)
    top_types = subset["incident_type"].value_counts().head(3).to_dict()
    top_subtypes = subset["incident_subtype"].fillna("").replace("", pd.NA).dropna().value_counts().head(3).to_dict()
    placeholder = (
        "Placeholder (agent call commented out): "
        f"Top incident categories here are {list(top_types.keys())}. "
        f"Most common subtype signals are {list(top_subtypes.keys()) if top_subtypes else ['insufficient subtype detail']}. "
        "Suggested actions: 1) triage high/critical open incidents within 24h, "
        "2) launch staffing/resource check for repeated subtype patterns, "
        "3) assign owner and SLA for closures."
    )
    st.text_area("Trend Summary", value=placeholder, height=120, disabled=True)

    st.markdown("**Incident Type Distribution**")
    st.bar_chart(subset["incident_type"].value_counts())
    st.markdown("**Subtype Distribution**")
    subtype_counts = subset["incident_subtype"].fillna("").replace("", pd.NA).dropna().value_counts().head(12)
    if not subtype_counts.empty:
        st.bar_chart(subtype_counts)

    st.markdown("**Top Issue Clusters (model-derived, ~10 groups)**")
    clusters = infer_top_issue_clusters(subset, n_clusters=10)
    if not clusters.empty:
        st.dataframe(clusters, use_container_width=True)
    else:
        st.info("Not enough text to cluster incident issues yet.")

    st.markdown("**Recent Incident Log**")
    view_cols = [
        "timestamp_utc",
        "incident_scope",
        "incident_type",
        "incident_subtype",
        "severity",
        "status",
        "description",
    ]
    st.dataframe(subset.sort_values("timestamp_utc", ascending=False)[view_cols], use_container_width=True)


def main():
    st.title("SafeMD.ai - Verified Ground Truth Intelligence")
    st.caption("Real-time capability verification, equity mapping, and DMAIC decision support")

    uploaded = None
    with st.expander("Advanced: override local pipeline output"):
        uploaded = st.file_uploader("Upload pipeline output JSON", type=["json"])
        st.caption("You can ignore this in normal use. By default, the app reads local `databricks_agent_output.json`.")
    data = load_data(uploaded, Path("databricks_agent_output.json"))
    facility_rows = data.get("facility_results", [])
    dmaic = data.get("dmaic_summary", {})
    equity = data.get("equity_summary", {})

    incidents_path = Path("ui/incidents.jsonl")
    incidents_df = load_incidents(incidents_path)
    df = to_df(facility_rows)
    if not df.empty:
        df["healthcare_types"] = df.apply(lambda r: classify_healthcare_types(r.to_dict()), axis=1)
        if len(df) < 1000:
            st.warning(
                "Current facility dataset has fewer than 1000 records. "
                "For full-scale demo, regenerate `databricks_agent_output.json` with a higher `--max-records` value."
            )
    dashboard_metrics(df, incidents_df)

    tab_map, tab_report, tab_analysis, tab_dashboard = st.tabs(
        ["Live Map", "Incident Reporting", "Incident Analysis", "Executive Dashboard"]
    )

    with tab_map:
        if not df.empty:
            filtered_df = apply_filters(df, incidents_df)
            facility_types = sorted(filtered_df["facility_type_id"].dropna().unique().tolist())
            selected_type = st.selectbox("Facility Type (facilityTypeId)", ["All"] + facility_types)
            if selected_type != "All":
                filtered_df = filtered_df[filtered_df["facility_type_id"] == selected_type]
            st.write(f"Showing {len(filtered_df):,} of {len(df):,} facilities")
            render_map(filtered_df, incidents_df)
            render_facility_detail(filtered_df)
        else:
            st.info("No facility data loaded.")

    with tab_report:
        render_incident_reporting_tab(df, incidents_path)

    with tab_analysis:
        render_incident_analysis_tab(df, incidents_df)

    with tab_dashboard:
        render_dmaic(dmaic)
        st.subheader("Equity Summary")
        if equity:
            st.json(equity)
        else:
            st.info("No equity summary available.")


if __name__ == "__main__":
    main()
