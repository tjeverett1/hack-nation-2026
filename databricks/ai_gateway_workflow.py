"""
Databricks AI Gateway workflow for the healthcare-facility verification project.

This script implements:
- Step A: Intake/Cleansing + semantic retrieval (Triage Agent)
- Step B: Claim extraction + validation loop (Reasoning + Validator)
- Step C: Schema alignment (IDP layer)
- Step D: DMAIC reporting payload generation (Reporting Agent)
- Step E: Policy/geospatial sync payload generation (Equity Agent)

Free API strategy (build stage):
- Point `PRIMARY_LLM_ENDPOINT` and `VALIDATOR_LLM_ENDPOINT` at Databricks AI Gateway
  routes backed by free-tier models (for example: Groq Llama 3.1/3.3).

Recommended production API after build stage:
- Anthropic Claude Sonnet (strong reasoning/verification quality for clinical text).
  Use a dedicated Databricks External Model endpoint for reliability and governance.
"""

from __future__ import annotations

import ast
import argparse
import json
import os
from dataclasses import dataclass
from math import isnan
from typing import Any

import pandas as pd
import requests


# -----------------------
# Runtime configuration
# -----------------------
DBR_HOST = os.environ.get("DATABRICKS_HOST", "")
DBR_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

VECTOR_ENDPOINT = os.environ.get("VECTOR_ENDPOINT", "mosaic-ai-vector-search")
PRIMARY_LLM_ENDPOINT = os.environ.get("PRIMARY_LLM_ENDPOINT", "gateway-llm-free")
VALIDATOR_LLM_ENDPOINT = os.environ.get("VALIDATOR_LLM_ENDPOINT", "gateway-llm-free")
REPORTING_LLM_ENDPOINT = os.environ.get("REPORTING_LLM_ENDPOINT", "gateway-llm-free")

GOLDEN_HOUR_RADIUS_KM = float(os.environ.get("GOLDEN_HOUR_RADIUS_KM", "80"))
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", "5"))


@dataclass
class FacilityRecord:
    row_id: int
    name: str
    city: str
    state: str
    pincode: str
    latitude: float | None
    longitude: float | None
    triage_text: str
    raw: dict[str, Any]


class DatabricksServingClient:
    def __init__(self, host: str, token: str):
        if not host or not token:
            raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN must be set.")
        self.host = host.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def invoke(self, endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host}/serving-endpoints/{endpoint_name}/invocations"
        resp = requests.post(url, headers=self.headers, json=payload, timeout=120)
        resp.raise_for_status()
        if resp.text:
            return resp.json()
        return {}


class LocalMockClient:
    """Offline mock client for local workflow testing."""

    def invoke(self, endpoint_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if "query_text" in payload:
            return {"matches": [{"score": 0.92, "snippet": "Mock historical facility similarity context."}]}

        messages = payload.get("messages", [])
        user_text = messages[-1]["content"] if messages else ""
        if "Return JSON with:" in user_text and "claims:" in user_text:
            return {
                "claims": [
                    {
                        "claim": "functional ICU",
                        "evidence_snippet": "ICU mentioned in descriptive text.",
                        "confidence_0_to_1": 0.66,
                    }
                ],
                "inferred_services": ["critical care", "emergency medicine"],
                "inferred_risks": ["possible staffing inconsistency"],
                "missing_critical_fields": ["24/7 backup power status", "ventilator count"],
                "staffing_signals": [
                    {
                        "role": "anesthesiologist",
                        "status": "VISITING",
                        "evidence_snippet": "Visiting specialist available.",
                    }
                ],
                "acuity_units": ["ICU"],
            }
        if "Return JSON:" in user_text and "trust_score" in user_text:
            return {
                "validations": [
                    {
                        "claim": "functional ICU",
                        "verdict": "PARTIAL",
                        "evidence_snippet": "ICU claim present but support infra not explicit.",
                        "notes": "No explicit mention of ventilators or power backup.",
                    }
                ],
                "trust_score": 72.0,
                "contradiction_count": 0,
                "recommended_human_review": False,
                "anomaly_flags": [],
                "chain_of_thought_summary": (
                    "ICU claim found; confidence reduced due to missing infrastructure corroboration."
                ),
            }
        if "Schema (target):" in user_text:
            return {
                "facility_name": "Mock Facility",
                "location": {"city": "Unknown", "state": "Unknown", "pincode": "UNKNOWN", "latitude": None, "longitude": None},
                "confirmed_capabilities": ["outpatient services"],
                "unverified_claims": ["functional ICU"],
                "missing_fields": ["24/7 backup power status", "ventilator count"],
                "staffing_signals": [{"role": "anesthesiologist", "status": "VISITING", "evidence_snippet": "Visiting specialist available."}],
                "verification": {
                    "trust_score": 72.0,
                    "requires_human_review": False,
                    "anomaly_flags": [],
                    "source_evidence": [{"claim": "functional ICU", "snippet": "ICU in description; infra unknown."}],
                },
            }
        if "Return JSON:" in user_text and "define:" in user_text:
            return {
                "define": "Regional care-access and functional-capacity inconsistencies.",
                "measure": [{"metric": "avg_trust_score", "value": 72.0, "interpretation": "Moderate confidence baseline."}],
                "analyze": ["Primary uncertainty driven by missing equipment/staffing detail."],
                "improve": ["Prioritize one-resource-away facilities and collect structured updates."],
                "control": ["Weekly trust trend review + correction-loop monitoring."],
                "district_risk_signals": [{"district": "Unknown", "risk_level": "medium", "reason": "Sparse verification evidence."}],
            }
        return {}


def _parse_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                return []
    return []


def _unknown(value: Any) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float) and pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def build_triage_text(row: pd.Series) -> str:
    specialties = ", ".join(_parse_list(row.get("specialties")))
    procedures = ", ".join(_parse_list(row.get("procedure")))
    equipment = ", ".join(_parse_list(row.get("equipment")))
    capability = ", ".join(_parse_list(row.get("capability")))

    # The "UNKNOWN" markers reduce hallucination risk and preserve missingness signal.
    blocks = [
        f"Facility Name: {_unknown(row.get('name'))}",
        f"Type: {_unknown(row.get('facilityTypeId'))}",
        f"Operator: {_unknown(row.get('operatorTypeId'))}",
        f"Address: {_unknown(row.get('address_line1'))}, {_unknown(row.get('address_city'))}, {_unknown(row.get('address_stateOrRegion'))}, {_unknown(row.get('address_zipOrPostcode'))}",
        f"Description: {_unknown(row.get('description'))}",
        f"Specialties: {_unknown(specialties)}",
        f"Procedures: {_unknown(procedures)}",
        f"Equipment: {_unknown(equipment)}",
        f"Capability: {_unknown(capability)}",
        f"Doctors: {_unknown(row.get('numberDoctors'))}",
        f"Capacity: {_unknown(row.get('capacity'))}",
    ]
    return "\n".join(blocks)


def load_and_prepare_records(csv_path: str) -> list[FacilityRecord]:
    df = pd.read_csv(csv_path, low_memory=False)
    records: list[FacilityRecord] = []
    for idx, row in df.iterrows():
        records.append(
            FacilityRecord(
                row_id=int(idx),
                name=_unknown(row.get("name")),
                city=_unknown(row.get("address_city")),
                state=_unknown(row.get("address_stateOrRegion")),
                pincode=_unknown(row.get("address_zipOrPostcode")),
                latitude=pd.to_numeric(row.get("latitude"), errors="coerce"),
                longitude=pd.to_numeric(row.get("longitude"), errors="coerce"),
                triage_text=build_triage_text(row),
                raw={k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
            )
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeMD.ai Databricks workflow runner")
    parser.add_argument("csv_path", help="Input facilities CSV")
    parser.add_argument("--output-path", default="databricks_agent_output.json", help="Output JSON file")
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "databricks"],
        help="local = no credentials required (mocked calls); databricks = real endpoint invocations",
    )
    parser.add_argument("--max-records", type=int, default=1500, help="Limit processed rows for faster local tests")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
        if isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def triage_retrieve_context(
    client: DatabricksServingClient, record: FacilityRecord, top_k: int = TOP_K_RETRIEVAL
) -> list[dict[str, Any]]:
    payload = {
        "query_text": record.triage_text,
        "top_k": top_k,
        "filters": {"address_country": "India"},
    }
    response = client.invoke(VECTOR_ENDPOINT, payload)
    return response.get("matches", [])


def _chat_payload(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1800,
    }


def extract_claims(
    client: DatabricksServingClient, record: FacilityRecord, context_matches: list[dict[str, Any]]
) -> dict[str, Any]:
    system_prompt = (
        "You are SafeMD.ai Triage/Reasoning Agent for Indian healthcare records. "
        "Extract explicit claims only from evidence text. Handle Indian shorthand/dialect variants "
        "(e.g., O2, NICU, OT, HDU) conservatively. Separate resident specialist vs visiting consultant. "
        "Ignore boilerplate marketing text. Output JSON only."
    )
    user_prompt = f"""
Facility record:
{record.triage_text}

Nearest contextual matches:
{json.dumps(context_matches, ensure_ascii=False)[:12000]}

Return JSON with:
- claims: [{{claim, evidence_snippet, confidence_0_to_1}}]
- inferred_services: [string]
- inferred_risks: [string]
- missing_critical_fields: [string]
- staffing_signals: [{{role, status: RESIDENT|VISITING|UNKNOWN, evidence_snippet}}]
- acuity_units: [string]   // e.g., ICU, Trauma, Oncology, Dialysis
"""
    return client.invoke(PRIMARY_LLM_ENDPOINT, _chat_payload(system_prompt, user_prompt))


def validate_claims(
    client: DatabricksServingClient,
    record: FacilityRecord,
    context_matches: list[dict[str, Any]],
    extracted: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict SafeMD.ai Validator Agent. Confirm or contradict each claim using only text evidence. "
        "Run a second-pass contradiction scan even for high-confidence claims. "
        "Flag suspicious 'perfect but vague' records for audit. Output JSON only."
    )
    user_prompt = f"""
Facility record:
{record.triage_text}

Context:
{json.dumps(context_matches, ensure_ascii=False)[:12000]}

Claims to validate:
{json.dumps(extracted, ensure_ascii=False)[:8000]}

Return JSON:
- validations: [{{claim, verdict: SUPPORTED|PARTIAL|UNSUPPORTED, evidence_snippet, notes}}]
- trust_score: float (0-100)
- contradiction_count: int
- recommended_human_review: bool
- anomaly_flags: [string]
- chain_of_thought_summary: string
"""
    return client.invoke(VALIDATOR_LLM_ENDPOINT, _chat_payload(system_prompt, user_prompt))


def align_to_vf_schema(
    client: DatabricksServingClient,
    record: FacilityRecord,
    extracted: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    # Replace this schema snippet with the exact Virtue Foundation Pydantic model fields.
    schema_hint = {
        "facility_name": "string",
        "location": {
            "city": "string",
            "state": "string",
            "pincode": "string",
            "latitude": "float|null",
            "longitude": "float|null",
        },
        "confirmed_capabilities": ["string"],
        "unverified_claims": ["string"],
        "missing_fields": ["string"],
        "verification": {
            "trust_score": "float",
            "requires_human_review": "bool",
            "anomaly_flags": ["string"],
            "source_evidence": [{"claim": "string", "snippet": "string"}],
        },
    }
    system_prompt = (
        "You are a schema mapper. Return strictly valid JSON matching the provided schema."
    )
    user_prompt = f"""
Schema (target):
{json.dumps(schema_hint)}

Source:
- record: {json.dumps(record.raw, ensure_ascii=False)[:9000]}
- extracted: {json.dumps(extracted, ensure_ascii=False)[:9000]}
- validated: {json.dumps(validated, ensure_ascii=False)[:9000]}
"""
    return client.invoke(PRIMARY_LLM_ENDPOINT, _chat_payload(system_prompt, user_prompt))


def heuristic_anomaly_flags(record: FacilityRecord, validated: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    trust = _safe_float(validated.get("trust_score"))
    contradiction_count = int(validated.get("contradiction_count", 0) or 0)
    triage = record.triage_text.lower()
    sparse = triage.count("unknown")
    has_icu_claim = "icu" in triage
    has_supporting_terms = any(
        term in triage
        for term in ["ventilator", "24/7 power", "power backup", "nursing", "critical care"]
    )
    if trust is not None and trust >= 95 and sparse >= 5:
        flags.append("HIGH_TRUST_LOW_DETAIL")
    if contradiction_count > 0:
        flags.append("INTERNAL_CONTRADICTION")
    if has_icu_claim and not has_supporting_terms:
        flags.append("ICU_WITHOUT_SUPPORTING_INFRA")
    return flags


def generate_dmaic_summary(client: DatabricksServingClient, schema_rows: list[dict[str, Any]]) -> dict[str, Any]:
    system_prompt = (
        "You are a Lean healthcare operations analyst. Build DMAIC summary and A3-style outputs in JSON."
    )
    user_prompt = f"""
Structured facilities:
{json.dumps(schema_rows[:200], ensure_ascii=False)[:15000]}

Return JSON:
- define: string
- measure: [{{metric, value, interpretation}}]
- analyze: [string]
- improve: [string]
- control: [string]
- district_risk_signals: [{{district, risk_level, reason}}]
"""
    return client.invoke(REPORTING_LLM_ENDPOINT, _chat_payload(system_prompt, user_prompt))


def build_ui_payload(
    record: FacilityRecord,
    aligned: dict[str, Any],
    validated: dict[str, Any],
) -> dict[str, Any]:
    return {
        "facility_id": record.row_id,
        "facility_name": record.name,
        "facility_type_id": record.raw.get("facilityTypeId", "UNKNOWN"),
        "city": record.city,
        "state": record.state,
        "pincode": record.pincode,
        "coordinates": {"lat": record.latitude, "lon": record.longitude},
        "trust_score": validated.get("trust_score"),
        "needs_human_review": validated.get("recommended_human_review", False),
        "anomaly_flags": validated.get("anomaly_flags", []),
        "reasoning_trace": validated.get("chain_of_thought_summary", ""),
        "confirmed_capabilities": aligned.get("confirmed_capabilities", []),
        "unverified_claims": aligned.get("unverified_claims", []),
        "missing_fields": aligned.get("missing_fields", []),
        "staffing_signals": aligned.get("staffing_signals", []),
        "evidence": validated.get("validations", []),
        "urgent_need": len(aligned.get("missing_fields", [])) > 0 and (validated.get("trust_score", 0) or 0) >= 50,
        "one_resource_away": len(aligned.get("missing_fields", [])) == 1,
        # Frontend action hook for correction feedback loop.
        "ui_actions": {"allow_correction_submit": True, "correction_endpoint_key": "triage_feedback_ingest"},
    }


def geospatial_gap_summary(facility_rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Placeholder output for UC function enrichment stage.
    # In Databricks SQL, this should be upgraded with ai_query + geospatial joins.
    high_risk = [r for r in facility_rows if (r.get("trust_score") or 0) < 60 or r.get("urgent_need")]
    return {
        "golden_hour_radius_km": GOLDEN_HOUR_RADIUS_KM,
        "high_risk_facility_count": len(high_risk),
        "notes": "Join with census Delta tables + travel-time matrix for true desert mapping.",
    }


def run_pipeline(
    csv_path: str,
    output_path: str = "databricks_agent_output.json",
    mode: str = "local",
    max_records: int = 200,
) -> None:
    client = DatabricksServingClient(DBR_HOST, DBR_TOKEN) if mode == "databricks" else LocalMockClient()
    records = load_and_prepare_records(csv_path)
    if max_records > 0:
        records = records[:max_records]

    final_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []

    for record in records:
        ctx = triage_retrieve_context(client, record, TOP_K_RETRIEVAL)
        extracted = extract_claims(client, record, ctx)
        validated = validate_claims(client, record, ctx, extracted)
        validated["anomaly_flags"] = sorted(
            set(validated.get("anomaly_flags", []) + heuristic_anomaly_flags(record, validated))
        )
        aligned = align_to_vf_schema(client, record, extracted, validated)
        schema_rows.append(aligned)
        final_rows.append(build_ui_payload(record, aligned, validated))

    dmaic = generate_dmaic_summary(client, schema_rows)
    equity = geospatial_gap_summary(final_rows)
    output = {
        "config": {"golden_hour_radius_km": GOLDEN_HOUR_RADIUS_KM, "top_k": TOP_K_RETRIEVAL},
        "facility_results": final_rows,
        "dmaic_summary": dmaic,
        "equity_summary": equity,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(final_rows)} facility records to {output_path} (mode={mode})")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        csv_path=args.csv_path,
        output_path=args.output_path,
        mode=args.mode,
        max_records=args.max_records,
    )
