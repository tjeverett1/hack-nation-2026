import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="SafeMD.ai", page_icon="🏥", layout="wide")

# Live map → Facility Intelligence (selectbox) sync when a map point is selected.
MAP_FOCUS_FACILITY_ID = "map_focus_facility_id"
FACILITY_INTEL_SELECT_KEY = "live_map_facility_intel_pick"
# Avoid re-queueing map focus on every rerun while Plotly keeps the same point selected.
LIVE_MAP_LAST_APPLIED_FID = "_live_map_last_applied_selection_fid"

# Extra horizontal space between tab triggers (Streamlit tabs are tight by default).
TABS_SPACING_CSS = """
<style>
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 2.75rem !important;
        justify-content: flex-start !important;
        flex-wrap: wrap !important;
        row-gap: 0.75rem !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding-left: 1.35rem !important;
        padding-right: 1.35rem !important;
        min-width: 9rem;
        justify-content: center;
    }
</style>
"""


def inject_tab_spacing_css() -> None:
    st.markdown(TABS_SPACING_CSS, unsafe_allow_html=True)


SEVERITY_GUIDANCE = {
    "Unexpected Outcome": {
        "low": "Minor post-injection bruising; mild nausea from prescribed medication.",
        "medium": "Allergic rash requiring antihistamines; delayed wound healing requiring extra follow-up.",
        "high": "Hospital-acquired infection requiring IV antibiotics; minor permanent nerve damage post-surgery.",
        "critical": "Unexplained maternal mortality; permanent disability due to misdiagnosed stroke.",
    },
    "Insufficient Resources": {
        "low": "Temporary shortage of adhesive bandages; limited generic Paracetamol brand options.",
        "medium": "Out of stock pediatric antibiotics; lack of sterile dressing kits for wound care.",
        "high": "Blood bank depleted of O+/A+; lack of dialysis kits for a scheduled patient.",
        "critical": "Zero oxygen supply in high-acuity ward; no anti-venom in high-risk snakebite zone.",
    },
    "Insufficient Staff": {
        "low": "Delayed discharge due to admin shortage; janitorial delay in non-clinical areas.",
        "medium": "One nurse managing 20+ low-acuity patients; no lab technician for routine blood work.",
        "high": "No surgeon for emergency appendectomy; no trained neonatal nurse for premature birth.",
        "critical": "No MD doctor during night emergency; no anesthesiologist during active C-section.",
    },
    "Equipment Failure": {
        "low": "Dead batteries in digital thermometer; worn velcro on manual BP cuff.",
        "medium": "ECG malfunction causing 2-hour delay; broken autoclave for non-surgical tools.",
        "high": "Ventilator malfunction while intubated patient is on support; X-ray burnout in trauma center.",
        "critical": "Defibrillator failure during cardiac arrest; central oxygen manifold total failure.",
    },
    "Infrastructure Failure": {
        "low": "Single light fixture out in hallway; broken waiting-room seating.",
        "medium": "Intermittent Wi-Fi blocking real-time records; HVAC failure in admin wing.",
        "high": "Water contamination in surgical wing; grid failure and generator fails to start.",
        "critical": "Cold-chain failure destroys vaccines/insulin; structural fire or ward collapse.",
    },
    "Near Miss": {
        "low": "Expired saline found before use; wrong patient name on chart caught early.",
        "medium": "Mislabeled lab sample caught pre-processing; incorrect dose caught by second nurse.",
        "high": "Wrong-limb surgery nearly initiated but caught in timeout; near-empty oxygen before transfer.",
        "critical": "Fire alarm failed during electrical fire (staff caught it); major medication mix-up caught at injection moment.",
    },
}


def load_data(uploaded_file, fallback_path: Path) -> dict:
    if uploaded_file is not None:
        return json.load(uploaded_file)
    if fallback_path.exists():
        return json.loads(fallback_path.read_text(encoding="utf-8"))
    return {"facility_results": [], "dmaic_summary": {}, "equity_summary": {}}


def normalize_facility_type_id(value: object) -> str:
    """Canonical facility_type labels (e.g. merge typo variants)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "UNKNOWN"
    s = str(value).strip()
    if not s:
        return "UNKNOWN"
    if s.lower() == "farmacy":
        return "pharmacy"
    return s


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
    df["facility_type_id"] = df["facility_type_id"].map(normalize_facility_type_id)
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


def _incident_impact_score(subset: pd.DataFrame) -> float:
    if subset.empty:
        return 0.0
    severity_w = {"low": 1.0, "medium": 2.0, "high": 4.0, "critical": 8.0}
    # Scope gets the highest weight by design.
    scope_w = {
        "Single Patient": 1.0,
        "Single Department": 2.5,
        "Whole Hospital": 8.0,
        "Regional Network": 10.0,
        "Supply Chain": 5.0,
        "Staff Safety": 4.0,
    }
    status_w = {"open": 1.0, "monitoring": 0.6, "resolved": 0.2}
    sev = subset["severity"].fillna("medium").map(severity_w).fillna(2.0)
    scp = subset["incident_scope"].fillna("Single Patient").map(scope_w).fillna(1.0)
    sts = subset["status"].fillna("open").map(status_w).fillna(1.0)
    return float((sev * scp * sts).sum())


def incident_impact_for_facility(incidents_df: pd.DataFrame, facility_id: int) -> tuple[float, bool]:
    if incidents_df.empty:
        return 0.0, False
    subset = incidents_df[incidents_df["facility_id"] == facility_id].copy()
    score = _incident_impact_score(subset)
    has_power = subset["incident_subtype"].fillna("").astype(str).str.lower().str.contains("power outage").any()
    return score, bool(has_power)


def severity_weighted_incident_count_for_facility(incidents_df: pd.DataFrame, facility_id: int) -> float:
    if incidents_df.empty:
        return 0.0
    subset = incidents_df[incidents_df["facility_id"] == facility_id]
    if subset.empty:
        return 0.0
    severity_w = {"low": 1.0, "medium": 2.0, "high": 4.0, "critical": 8.0}
    sev = subset["severity"].fillna("medium").astype(str).str.lower().map(severity_w).fillna(2.0)
    return float(sev.sum())


def _gradient_color_from_value(value: float, min_value: float, max_value: float) -> list[int]:
    if max_value <= min_value:
        t = 0.0
    else:
        t = (value - min_value) / (max_value - min_value)
    t = float(np.clip(t, 0.0, 1.0))
    # Cool-to-hot gradient (blue -> red) while preserving existing dot style.
    r = int(round(60 + (235 - 60) * t))
    g = int(round(145 + (35 - 145) * t))
    b = int(round(220 + (30 - 220) * t))
    return [r, g, b, 185]


def build_incident_gradient_map_df(map_df: pd.DataFrame, incidents_df: pd.DataFrame) -> pd.DataFrame:
    working = map_df.copy()
    if working.empty:
        return working
    working["incident_weighted_count"] = working["facility_id"].apply(
        lambda fid: severity_weighted_incident_count_for_facility(incidents_df, int(fid))
    )
    min_v = float(working["incident_weighted_count"].min()) if not working.empty else 0.0
    max_v = float(working["incident_weighted_count"].max()) if not working.empty else 0.0
    colors = working["incident_weighted_count"].apply(
        lambda v: _gradient_color_from_value(float(v), min_v, max_v)
    )
    working["dot_r"] = colors.apply(lambda c: int(c[0]))
    working["dot_g"] = colors.apply(lambda c: int(c[1]))
    working["dot_b"] = colors.apply(lambda c: int(c[2]))
    working["dot_a"] = colors.apply(lambda c: int(c[3]))
    working["location_name"] = (
        working["facility_name"].fillna("Unknown facility").astype(str)
        + " ("
        + working["city"].fillna("Unknown city").astype(str)
        + ", "
        + working["state"].fillna("Unknown state").astype(str)
        + ")"
    )
    return working


def facility_status_for_map(incidents_df: pd.DataFrame, facility_id: int) -> str:
    if incidents_df.empty:
        return "NOMINAL"
    subset = incidents_df[incidents_df["facility_id"] == facility_id]
    if subset.empty:
        return "NOMINAL"
    open_rows = subset[subset["status"] == "open"]
    if not open_rows.empty:
        has_critical = (open_rows["severity"].fillna("").str.lower() == "critical").any()
        has_power = open_rows["incident_subtype"].fillna("").str.lower().str.contains("power outage").any()
        if has_critical or has_power:
            return "CRITICAL"
        return "ACTIVE"
    if (subset["status"] == "monitoring").any():
        return "MONITORING"
    return "NOMINAL"


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def render_executive_map_plotly_with_placeholder_box(
    base_plot: pd.DataFrame,
    mid_lat: float,
    mid_lon: float,
) -> None:
    """Plotly Mapbox + OSM tiles (reliable in Streamlit); red filled box = placeholder desert region."""
    dlat, dlon = 1.2, 1.2
    box_lon = [mid_lon - dlon, mid_lon + dlon, mid_lon + dlon, mid_lon - dlon, mid_lon - dlon]
    box_lat = [mid_lat - dlat, mid_lat - dlat, mid_lat + dlat, mid_lat + dlat, mid_lat - dlat]

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            mode="lines",
            lon=box_lon,
            lat=box_lat,
            fill="toself",
            fillcolor="rgba(255, 55, 55, 0.32)",
            line=dict(color="darkred", width=2),
            name="placeholder",
            hovertemplate="Desert overlay (placeholder)<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=base_plot["lon"],
            lat=base_plot["lat"],
            marker=dict(size=7, color="rgb(22, 105, 170)"),
            text=base_plot["tip"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=4.2,
        mapbox_center=dict(lat=mid_lat, lon=mid_lon),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})


def _plotly_point_as_dict(pt0) -> dict:
    if pt0 is None:
        return {}
    if isinstance(pt0, dict):
        return pt0
    out = {}
    for k in ("customdata", "point_index", "pointNumber", "curveNumber"):
        if hasattr(pt0, k):
            out[k] = getattr(pt0, k)
    return out


def _facility_id_from_plotly_map_event(event, plot_df: pd.DataFrame) -> int | None:
    """Read facility_id from Streamlit Plotly selection (customdata or point_index)."""
    if event is None:
        return None
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if sel is None:
        return None
    pts = getattr(sel, "points", None)
    if pts is None and isinstance(sel, dict):
        pts = sel.get("points")
    if not pts:
        return None
    pt0 = pts[0] if isinstance(pts, (list, tuple)) else None
    pt0 = _plotly_point_as_dict(pt0)
    cd = pt0.get("customdata")
    if cd is not None:
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            try:
                return int(cd[0])
            except (TypeError, ValueError):
                pass
        try:
            return int(cd)
        except (TypeError, ValueError):
            pass
    idx = pt0.get("point_index")
    if idx is None:
        idx = pt0.get("pointNumber")
    if idx is not None:
        ii = int(idx)
        if 0 <= ii < len(plot_df):
            return int(plot_df.iloc[ii]["facility_id"])
    return None


def render_live_ground_truth_facility_map_plotly(map_df: pd.DataFrame, incidents_df: pd.DataFrame) -> None:
    """Plotly + OSM tiles; dot color scales with incident report volume at each facility."""
    plot_df = map_df.copy()
    plot_df["facility_id"] = pd.to_numeric(plot_df["facility_id"], errors="coerce")
    plot_df = plot_df.dropna(subset=["facility_id"])
    plot_df["facility_id"] = plot_df["facility_id"].astype(int)

    if incidents_df.empty:
        plot_df["incident_reports"] = 0
        plot_df["incident_weighted"] = 0.0
    else:
        plot_df["incident_reports"] = plot_df["facility_id"].apply(
            lambda fid: int((incidents_df["facility_id"] == int(fid)).sum())
        )
        plot_df["incident_weighted"] = plot_df["facility_id"].apply(
            lambda fid: severity_weighted_incident_count_for_facility(incidents_df, int(fid))
        )

    plot_df["facility_name"] = plot_df.get("facility_name", pd.Series(dtype=object)).fillna("").astype(str)
    plot_df["city"] = plot_df.get("city", pd.Series(dtype=object)).fillna("").astype(str)
    plot_df["state"] = plot_df.get("state", pd.Series(dtype=object)).fillna("").astype(str)
    plot_df["pincode"] = plot_df.get("pincode", pd.Series(dtype=object)).fillna("").astype(str)
    plot_df["facility_type_id"] = plot_df.get("facility_type_id", pd.Series(dtype=object)).fillna("").astype(str)
    plot_df["trust_score"] = pd.to_numeric(plot_df.get("trust_score"), errors="coerce")
    plot_df["map_label"] = plot_df["facility_name"].astype(str).str.strip().replace("", "—")

    mid_lat = float(plot_df["lat"].median())
    mid_lon = float(plot_df["lon"].median())

    z = plot_df["incident_reports"].astype(float)
    min_v = float(z.min())
    max_v = float(z.max())
    if max_v <= min_v:
        cmax = min_v + 1.0
    else:
        cmax = max_v

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=plot_df["lon"],
            lat=plot_df["lat"],
            marker=dict(
                size=9,
                color=z,
                colorscale=[
                    [0.0, "rgb(56, 130, 210)"],
                    [0.45, "rgb(120, 190, 210)"],
                    [0.72, "rgb(235, 195, 80)"],
                    [1.0, "rgb(220, 45, 35)"],
                ],
                cmin=min_v,
                cmax=cmax,
                colorbar=dict(title="Incident reports", tickformat=".0f"),
                showscale=True,
            ),
            text=plot_df["map_label"],
            customdata=np.expand_dims(plot_df["facility_id"].astype(int).values, axis=1),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=4.2,
        mapbox_center=dict(lat=mid_lat, lon=mid_lon),
        margin=dict(l=0, r=48, t=0, b=0),
        height=520,
        showlegend=False,
    )
    plotly_kwargs = dict(use_container_width=True, config={"scrollZoom": True, "displaylogo": False})
    try:
        event = st.plotly_chart(
            fig,
            key="live_map_facilities",
            on_select="rerun",
            selection_mode="points",
            **plotly_kwargs,
        )
    except TypeError:
        event = None
        st.plotly_chart(fig, **plotly_kwargs)

    fid = _facility_id_from_plotly_map_event(event, plot_df)
    if fid is not None and fid != st.session_state.get(LIVE_MAP_LAST_APPLIED_FID):
        st.session_state[MAP_FOCUS_FACILITY_ID] = fid
        st.session_state[LIVE_MAP_LAST_APPLIED_FID] = fid
        mask = plot_df["facility_id"] == fid
        name = str(plot_df.loc[mask, "map_label"].iloc[0]) if bool(mask.any()) else str(fid)
        try:
            st.toast(f"Opening: {name}")
        except Exception:
            pass

    st.caption(
        "Hover shows **facility name only**. **Click** a point to open it in Facility Intelligence below. "
        "Cooler dots = fewer incident reports; warmer = more (see color bar)."
    )


def compute_desert_overlay_points(
    all_df: pd.DataFrame,
    selected_facility_type: str,
    golden_hour_radius_km: float = 100.0,
    min_trust: float = 70.0,
    sample_n: int = 500,
) -> pd.DataFrame:
    if all_df.empty or selected_facility_type == "All":
        return pd.DataFrame(columns=["lat", "lon", "desert_strength"])
    base = all_df.copy()
    coords = pd.json_normalize(base["coordinates"]).rename(columns={"lat": "lat", "lon": "lon"})
    base = pd.concat([base.reset_index(drop=True), coords], axis=1).dropna(subset=["lat", "lon"])
    if base.empty:
        return pd.DataFrame(columns=["lat", "lon", "desert_strength"])

    targets = base[
        (base["facility_type_id"] == selected_facility_type) & (base["trust_score"].fillna(0) >= min_trust)
    ][["lat", "lon"]].to_numpy()
    sampled = base.sample(n=min(sample_n, len(base)), random_state=42)[["lat", "lon"]].copy()
    if len(targets) == 0:
        sampled["desert_strength"] = 1.0
        return sampled

    strengths = []
    for _, row in sampled.iterrows():
        dists = _haversine_km(row["lat"], row["lon"], targets[:, 0], targets[:, 1])
        min_dist = float(np.min(dists))
        strengths.append(max(0.0, min(1.0, (min_dist - golden_hour_radius_km) / golden_hour_radius_km)))
    sampled["desert_strength"] = strengths
    return sampled[sampled["desert_strength"] > 0].copy()


def get_facility_text_blob(row: dict) -> str:
    return " ".join(
        [
            " ".join(row.get("confirmed_capabilities", []) or []),
            " ".join(row.get("unverified_claims", []) or []),
            str(row.get("reasoning_trace", "")),
            str(row.get("facility_name", "")),
        ]
    ).lower()


def synthetic_trust_history(current: float, facility_id: int, n: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng((facility_id * 7919 + int(current * 10)) % (2**32))
    noise = rng.normal(0, 1.8, n)
    walk = np.clip(np.cumsum(noise) + float(current), 5, 100)
    walk[-1] = float(current)
    # Weekly points, oldest → newest; rightmost = today's date (UTC calendar day).
    end = pd.Timestamp.now(tz=timezone.utc).normalize()
    dates = [end - pd.Timedelta(weeks=(n - 1 - i)) for i in range(n)]
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "trust": walk})


@st.cache_data(show_spinner=False)
def get_dummy_api_ranked_results(selected_facility_type: str, n: int = 200) -> pd.DataFrame:
    # Supported contract formats:
    # 1) Simple:
    #    {"results":[{"place_name":"...", "specialty":"...", "score":87.2, "facility_type_id":"clinic"}]}
    # 2) Future schema:
    #    {
    #      "request_id":"...",
    #      "metadata": {...},
    #      "ranked_facilities":[
    #        {"rank":1,"facility":{"name":"...","type":"clinic","specialties":[...]},
    #         "scores":{"composite_0_100":91.2}, "distance_km":3.4, "incident_open_count":2, "reasons":[...]}
    #      ]
    #    }
    contract_paths = [Path("ui/dummy_api_response_v2.json"), Path("ui/dummy_api_response.json")]
    payload = None
    for path in contract_paths:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            break

    if payload is not None:
        rows = []
        if "ranked_facilities" in payload:
            for r in payload.get("ranked_facilities", []):
                facility = r.get("facility", {})
                score_obj = r.get("scores", {})
                rows.append(
                    {
                        "rank": r.get("rank"),
                        "place_name": facility.get("name"),
                        "specialty": ", ".join(facility.get("specialties", [])) if isinstance(facility.get("specialties"), list) else facility.get("specialties"),
                        "score": score_obj.get("score", score_obj.get("composite_0_100")),
                        "facility_type_id": normalize_facility_type_id(facility.get("type")),
                        "distance_km": r.get("distance_km"),
                        "incident_open_count": r.get("incident_open_count"),
                        "reasons": "; ".join(r.get("reasons", [])) if isinstance(r.get("reasons"), list) else r.get("reasons"),
                    }
                )
        else:
            rows = payload.get("results", [])

        df = pd.DataFrame(rows)
        if not df.empty:
            if "facility_type_id" in df.columns:
                df["facility_type_id"] = df["facility_type_id"].map(normalize_facility_type_id)
            if "facility_type_id" in df.columns and selected_facility_type != "All":
                typed = df[df["facility_type_id"].astype(str) == str(selected_facility_type)].copy()
                if not typed.empty:
                    df = typed
            if "rank" not in df.columns:
                df = df.reset_index(drop=True)
                df["rank"] = df.index + 1
            if "score" not in df.columns and "score_0_100" in df.columns:
                df["score"] = df["score_0_100"]
            required_cols = ["rank", "place_name", "specialty", "score"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[required_cols + [c for c in df.columns if c not in required_cols]].copy()
            df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
            # API is assumed unordered; rank locally by score.
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            df["rank"] = df.index + 1
            return df.head(n)

    # Fallback if file is missing/empty.
    seed = abs(hash(selected_facility_type)) % (2**32)
    rng = np.random.default_rng(seed)
    rows = [
        {
            "rank": i + 1,
            "place_name": f"Fallback Facility {selected_facility_type}-{i+1:03d}",
            "specialty": "General Healthcare",
            "score": float(np.round(rng.uniform(25, 99), 1)),
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


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
    clean_text = text[text.str.strip() != ""]
    if clean_text.empty:
        return pd.DataFrame(columns=["cluster_id", "cluster_label", "count"])
    # Guard for tiny sample sizes (e.g., 1 incident): k-means requires n_samples >= n_clusters.
    if len(clean_text) < 2:
        fallback_label = clean_text.iloc[0][:80] if len(clean_text.iloc[0]) > 0 else "single_issue"
        return pd.DataFrame(
            [{"cluster_id": 0, "cluster_label": fallback_label, "count": int(len(clean_text))}]
        )
    k = min(n_clusters, len(clean_text))
    vectorizer = TfidfVectorizer(stop_words="english", max_features=800)
    X = vectorizer.fit_transform(clean_text)
    if X.shape[1] == 0:
        return pd.DataFrame(
            [{"cluster_id": 0, "cluster_label": "insufficient_text_signal", "count": int(len(clean_text))}]
        )
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


def render_map(
    filtered_df: pd.DataFrame,
    incidents_df: pd.DataFrame,
):
    st.subheader("Live Ground Truth Facility Map")
    map_df = filtered_df.copy()
    if map_df.empty:
        st.info("No records match current filters.")
        return
    coords = pd.json_normalize(map_df["coordinates"]).rename(columns={"lat": "lat", "lon": "lon"})
    map_df = pd.concat([map_df.reset_index(drop=True), coords], axis=1)
    map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
    map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
    map_df = map_df.dropna(subset=["lat", "lon"])
    if map_df.empty:
        st.info("No geocoded coordinates in current selection.")
        return
    try:
        render_live_ground_truth_facility_map_plotly(map_df, incidents_df)
    except Exception as exc:
        st.map(map_df[["lat", "lon"]], zoom=4)
        st.caption(f"Plotly map unavailable ({exc!s}); using basic map. Install plotly: `pip install plotly`.")


def render_dmaic(dmaic: dict, df: pd.DataFrame, incidents_df: pd.DataFrame):
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

    st.subheader("Executive Map + Desert Overlay")
    if not df.empty:
        base = df.copy()
        coords = pd.json_normalize(base["coordinates"]).rename(columns={"lat": "lat", "lon": "lon"})
        base = pd.concat([base.reset_index(drop=True), coords], axis=1)
        base["lat"] = pd.to_numeric(base["lat"], errors="coerce")
        base["lon"] = pd.to_numeric(base["lon"], errors="coerce")
        base = base.dropna(subset=["lat", "lon"])
        if not base.empty:
            if "facility_name" in base.columns:
                base["facility_name"] = base["facility_name"].fillna("").astype(str)
            else:
                base["facility_name"] = ""
            ov_col1, ov_col2, ov_col3 = st.columns([1.15, 1.35, 2.2])
            with ov_col1:
                show_desert_overlay = st.checkbox(
                    "Show desert overlay",
                    value=False,
                    help="Sampled locations weighted by distance to the nearest high-trust “anchor” facility type (golden-hour style heuristic).",
                    key="exec_show_desert_overlay",
                )
            facility_types_exec = sorted(df["facility_type_id"].dropna().unique().tolist())
            default_anchor = 0
            for i, t in enumerate(facility_types_exec):
                if str(t).lower() in ("hospital", "traumacenter", "trauma_center"):
                    default_anchor = i
                    break
            with ov_col2:
                desert_anchor_type = st.selectbox(
                    "Anchor facility type",
                    facility_types_exec,
                    index=min(default_anchor, len(facility_types_exec) - 1),
                    disabled=not show_desert_overlay,
                    help="Verified anchors = this type with trust ≥ threshold below.",
                    key="exec_desert_anchor_type",
                )
            with ov_col3:
                desert_gh_km = st.number_input(
                    "Golden-hour radius (km)",
                    min_value=20.0,
                    max_value=250.0,
                    value=80.0,
                    step=5.0,
                    disabled=not show_desert_overlay,
                    key="exec_desert_gh_km",
                )
                desert_min_trust = st.slider(
                    "Min trust for anchor",
                    min_value=0,
                    max_value=100,
                    value=70,
                    disabled=not show_desert_overlay,
                    key="exec_desert_min_trust",
                )

            if show_desert_overlay:
                mid_lat = float(base["lat"].median())
                mid_lon = float(base["lon"].median())
                base_plot = base.copy()
                base_plot["trust_score"] = pd.to_numeric(base_plot.get("trust_score"), errors="coerce")
                base_plot["tip"] = base_plot.apply(
                    lambda r: f"{str(r.get('facility_name', ''))[:48]} — trust {r.get('trust_score', '')}",
                    axis=1,
                )
                try:
                    render_executive_map_plotly_with_placeholder_box(base_plot, mid_lat, mid_lon)
                except Exception as exc:
                    st.map(base[["lat", "lon"]], zoom=4)
                    st.caption(f"Plotly map failed ({exc!s}); showing Streamlit map only. Run `pip install -r ui/requirements.txt`.")
                st.caption(
                    f"Placeholder desert: **semi-transparent red box** at map center (Plotly + OpenStreetMap). "
                    f"Next step: drive overlay from anchor **{desert_anchor_type}**, **{desert_gh_km:.0f} km**, trust ≥ **{desert_min_trust}**."
                )
            else:
                st.map(base[["lat", "lon"]], zoom=4)

    st.subheader("Specialty Gap Matrix + Desert Index")
    if not df.empty:
        population_threshold = st.number_input("Population Threshold (X)", min_value=10000, value=150000, step=10000)
        req_trauma = st.checkbox("Require functioning Trauma capability", value=True)
        req_maternal = st.checkbox("Require functioning Maternal capability", value=True)
        req_dialysis = st.checkbox("Require functioning Dialysis capability", value=False)
        grouped = df.groupby("state", dropna=False).agg(
            facilities=("facility_id", "count"),
            avg_trust=("trust_score", "mean"),
        ).reset_index().rename(columns={"state": "region"})
        grouped["estimated_population"] = grouped["facilities"] * 12000
        grouped["verified_functional_beds_proxy"] = grouped["facilities"] * (grouped["avg_trust"].fillna(0) / 100.0) * 10
        grouped["desert_index_0_100"] = (100 - (grouped["verified_functional_beds_proxy"] / grouped["estimated_population"] * 1000)).clip(0, 100)
        grouped["danger_zone"] = grouped["estimated_population"] > population_threshold
        rules = []
        if req_trauma:
            rules.append("trauma")
        if req_maternal:
            rules.append("maternal")
        if req_dialysis:
            rules.append("dialysis")
        if rules:
            reg_has_rules = {}
            for region, sdf in df.groupby("state", dropna=False):
                blob = " ".join(sdf.apply(lambda r: get_facility_text_blob(r.to_dict()), axis=1).tolist())
                reg_has_rules[region] = all(rule in blob for rule in rules)
            grouped["danger_zone"] = grouped["danger_zone"] | ~grouped["region"].map(reg_has_rules).fillna(False)
        st.dataframe(
            grouped.sort_values(["danger_zone", "desert_index_0_100"], ascending=[False, False]),
            use_container_width=True,
        )

    st.subheader("Acuity Filters (Life-Saving Capability Combinations)")
    acuity_text = st.text_input("Capability Query", value="ventilator 24/7 electricity")
    if acuity_text.strip() and not df.empty:
        terms = acuity_text.lower().split()
        matched = df[df.apply(lambda r: all(t in get_facility_text_blob(r.to_dict()) for t in terms), axis=1)]
        st.write(f"Facilities matching acuity query: {len(matched)}")
        if not matched.empty:
            st.dataframe(
                matched[["facility_name", "city", "state", "trust_score"]].sort_values("trust_score", ascending=False).head(30),
                use_container_width=True,
            )


def render_facility_detail(filtered_df: pd.DataFrame):
    st.markdown('<span id="facility-intelligence"></span>', unsafe_allow_html=True)
    st.subheader("Facility Intelligence + Source Traceability")
    if filtered_df.empty:
        st.info("No facility selected.")
        return
    search_term = st.text_input("Search Facility", placeholder="Type facility or city...")
    match = filtered_df.copy()
    if search_term.strip():
        sl = search_term.lower()
        match = match[
            match["facility_name"].fillna("UNKNOWN").str.lower().str.contains(sl, na=False)
            | match["city"].fillna("UNKNOWN").str.lower().str.contains(sl, na=False)
            | match["state"].fillna("UNKNOWN").str.lower().str.contains(sl, na=False)
        ]
    if match.empty:
        st.info("No facilities match search.")
        return
    options = (
        match["facility_name"].fillna("UNKNOWN")
        + " | "
        + match["city"].fillna("UNKNOWN")
        + ", "
        + match["state"].fillna("UNKNOWN")
    ).tolist()

    if MAP_FOCUS_FACILITY_ID in st.session_state:
        try:
            focus_fid = int(st.session_state.pop(MAP_FOCUS_FACILITY_ID))
        except (TypeError, ValueError):
            focus_fid = None
        if focus_fid is not None:
            hit = match[match["facility_id"] == focus_fid]
            if not hit.empty:
                lab = (
                    hit["facility_name"].fillna("UNKNOWN").astype(str)
                    + " | "
                    + hit["city"].fillna("UNKNOWN").astype(str)
                    + ", "
                    + hit["state"].fillna("UNKNOWN").astype(str)
                ).iloc[0]
                if lab in options:
                    st.session_state[FACILITY_INTEL_SELECT_KEY] = lab

    if FACILITY_INTEL_SELECT_KEY in st.session_state:
        cur = st.session_state.get(FACILITY_INTEL_SELECT_KEY)
        if cur not in options:
            del st.session_state[FACILITY_INTEL_SELECT_KEY]

    selected_label = st.selectbox("Select Facility", options, key=FACILITY_INTEL_SELECT_KEY)
    row = match.reset_index(drop=True).iloc[options.index(selected_label)].to_dict()

    st.markdown(f"### {row.get('facility_name', 'UNKNOWN')}")
    st.write(f"**Location:** {row.get('city', 'UNKNOWN')}, {row.get('state', 'UNKNOWN')} ({row.get('pincode', 'UNKNOWN')})")
    trust = float(row.get("trust_score", 0) or 0)
    trust_color = "#cc0000" if trust < 50 else ("#0057b8" if trust < 75 else "#1f7a1f")
    st.markdown(
        f"<div style='font-size: 30px; font-weight: 700; color: {trust_color};'>Trust Score: {trust:.1f}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Trust score history (synthetic trend, weekly)**")
    hist = synthetic_trust_history(trust, int(row.get("facility_id", 0) or 0))
    st.line_chart(hist.set_index("date")["trust"], height=180)
    st.write(f"**Reasoning Trace:** {row.get('reasoning_trace', '')}")
    st.markdown("**Confirmed Capabilities**")
    for x in row.get("confirmed_capabilities", []):
        st.write(f"- {x}")
    st.markdown("**Unverified Claims**")
    for x in row.get("unverified_claims", []):
        st.write(f"- {x}")

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
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
    recommended = "medium"
    subtype_hint = incident_subtype.lower()
    if any(k in subtype_hint for k in ["power outage", "oxygen", "defibrillator", "collapse", "no md", "no anesthesiologist"]):
        recommended = "critical"
    elif any(k in subtype_hint for k in ["ventilator", "blood", "no surgeon", "contamination"]):
        recommended = "high"
    elif any(k in subtype_hint for k in ["delay", "wifi", "ecg", "sample"]):
        recommended = "medium"
    else:
        recommended = "low"
    severity = st.select_slider("Severity", options=["low", "medium", "high", "critical"], value=recommended)
    st.caption(f"Suggested severity: **{recommended.upper()}** based on incident subtype keywords.")
    guidance = SEVERITY_GUIDANCE.get(incident_type, {})
    if guidance:
        with st.expander("Severity Examples for this Incident Type", expanded=False):
            for level in ["low", "medium", "high", "critical"]:
                st.write(f"**{level.title()}**: {guidance.get(level, '')}")
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
    selected_label = st.selectbox("Choose Hospital", ["All Hospitals"] + facility_options, key="analysis_facility")

    if selected_label == "All Hospitals":
        subset = incidents_df.copy()
        st.markdown("### All Hospitals")
    else:
        idx = facility_options.index(selected_label)
        facility_id = int(df.iloc[idx]["facility_id"])
        facility_name = df.iloc[idx]["facility_name"]
        subset = incidents_df[incidents_df["facility_id"] == facility_id].copy()
        st.markdown(f"### {facility_name}")
        if subset.empty:
            st.info("No incidents for this hospital.")
            return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Scope: Total", len(subset))
    col2.metric("Selected Scope: Open", int((subset["status"] == "open").sum()))
    col3.metric("Selected Scope: Critical", int((subset["severity"] == "critical").sum()))
    col4.metric("Global Open Incidents", int((incidents_df["status"] == "open").sum()))

    st.markdown("**Overview of Trends + Suggested Actionables**")
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
    inject_tab_spacing_css()
    st.title("Ground Truth Intelligence")
    st.caption("Real-time capability verification, equity mapping, and DMAIC decision support")

    data = load_data(None, Path("databricks_agent_output.json"))
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
            facility_types = sorted(df["facility_type_id"].dropna().unique().tolist())
            selected_type = st.selectbox("Facility Type", ["All"] + facility_types)
            patient_need = st.text_input(
                "Patient Need",
                placeholder="e.g. trauma stabilization, dialysis, neonatal emergency",
            )
            filtered_df = df.copy()
            if selected_type != "All":
                filtered_df = filtered_df[filtered_df["facility_type_id"] == selected_type]

            map_zip = st.text_input(
                "Postal code (ZIP / PIN)",
                placeholder="e.g. 500013 — leave empty for all in current facility type",
                key="live_map_pin_filter",
            )
            if map_zip.strip():
                z = map_zip.strip()
                filtered_df = filtered_df[filtered_df["pincode"].astype(str).str.startswith(z)]

            st.write(f"Showing {len(filtered_df):,} of {len(df):,} facilities")
            render_map(filtered_df, incidents_df)
            render_facility_detail(filtered_df)

            st.markdown("#### Ranked Facilities (Top 200)")
            api_df = get_dummy_api_ranked_results(selected_type, n=200)
            if patient_need.strip():
                need = patient_need.lower().strip()
                narrowed = api_df[
                    api_df["specialty"].astype(str).str.lower().str.contains(need, regex=False)
                    | api_df["place_name"].astype(str).str.lower().str.contains(need, regex=False)
                ]
                if not narrowed.empty:
                    api_df = narrowed
            display_cols = ["rank", "place_name", "specialty", "score"]
            st.dataframe(
                api_df[display_cols].sort_values("rank"),
                use_container_width=True,
                height=360,
                hide_index=True,
            )
        else:
            st.info("No facility data loaded.")

    with tab_report:
        render_incident_reporting_tab(df, incidents_path)

    with tab_analysis:
        render_incident_analysis_tab(df, incidents_df)

    with tab_dashboard:
        render_dmaic(dmaic, df, incidents_df)
        st.subheader("Equity Summary")
        if equity:
            st.json(equity)
        else:
            st.info("No equity summary available.")


if __name__ == "__main__":
    main()
