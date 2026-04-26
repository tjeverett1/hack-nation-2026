"""
SafeMD.ai - RAG pipeline (portable / runs OUTSIDE the Databricks workspace).

Differences vs. the in-workspace notebook version:

* Authentication is explicit.  Inside Databricks, both `mlflow.deployments`
  and `databricks.vector_search.client.VectorSearchClient` pick up the
  notebook-runner identity automatically.  From a laptop / CI runner / Lambda
  we have to tell them where the workspace is and which token to use.

  We do this via two environment variables:
        DATABRICKS_HOST    e.g. https://dbc-83de82b8-c00f.cloud.databricks.com
        DATABRICKS_TOKEN   personal access token (User Settings -> Developer)

  Put them in a `.env` file at the repo root - `pipeline.run_pipeline`
  loads them with python-dotenv before instantiating the pipeline.

* Everything else (prompts, agent flow, JSON-extraction helper, mock data
  fallback) is kept identical to the notebook so behaviour is unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlflow.deployments
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. Configuration & Constants
# ==========================================
TRUST_SCORE_MIN = 0
TRUST_SCORE_MAX = 100
DEFAULT_TEMPERATURE = 0.1

# Token Limits
MAX_TOKENS_RANKING = 1000
MAX_TOKENS_TRIAGE = 1000
MAX_TOKENS_REPORTING = 2000

# Search Configuration
SEARCH_RADIUS_KM = 200
TOP_K_RESULTS = 5
VECTOR_SEARCH_ENDPOINT = "vector-db-endpoint"
INDEX_NAME = "hack_nation.silver.facilities_embed"

# Model Endpoints
RANKING_MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TRIAGE_MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
REPORTING_MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


# ==========================================
# 2. Auth helpers
# ==========================================
def _ensure_databricks_env() -> tuple[str, str]:
    """Validate that the two env vars MLflow + VectorSearch need are set.

    Returns (host, token) for callers that want to pass them through
    explicitly (e.g. into VectorSearchClient kwargs)."""
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not token:
        raise RuntimeError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set.\n"
            "  - Put them in a .env file at the repo root, or export them "
            "in your shell.\n"
            "  - Example values:\n"
            "        DATABRICKS_HOST=https://dbc-xxxx.cloud.databricks.com\n"
            "        DATABRICKS_TOKEN=dapi...\n"
        )
    # MLflow's Databricks client reads from these env vars.  No further setup
    # needed - we just normalise the host (strip trailing slash).
    host = host.rstrip("/")
    os.environ["DATABRICKS_HOST"] = host
    return host, token


# ==========================================
# 3. Pipeline Definition
# ==========================================
class SafeMDPipeline:
    """RAG Pipeline for Agentic Healthcare Maps and QI Reporting."""

    def __init__(
        self,
        vector_search_endpoint: str = VECTOR_SEARCH_ENDPOINT,
        index_name: str = INDEX_NAME,
        ranking_model: str = RANKING_MODEL_ENDPOINT,
        triage_model: str = TRIAGE_MODEL_ENDPOINT,
        reporting_model: str = REPORTING_MODEL_ENDPOINT,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        host, token = _ensure_databricks_env()

        # MLflow Deployments client reuses DATABRICKS_HOST / DATABRICKS_TOKEN
        # from the environment when the target is "databricks".
        self._client = mlflow.deployments.get_deploy_client("databricks")

        # VectorSearchClient picks up the same env vars but we pass them
        # explicitly to avoid any ambiguity if the user is also signed in
        # to the Databricks CLI on this machine.
        self.vs_client = VectorSearchClient(
            workspace_url=host,
            personal_access_token=token,
        )

        # Same 384-dim model that built the index - DO NOT change without
        # rebuilding hack_nation.silver.facilities_embed.
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.vector_search_endpoint = vector_search_endpoint
        self.index_name = index_name
        self.ranking_model = ranking_model
        self.triage_model = triage_model
        self.reporting_model = reporting_model

    # --------------------------------------
    # Helpers
    # --------------------------------------
    def _parse_llm_json(self, raw_response: str) -> dict[str, Any]:
        """Extract a JSON object from a string that may contain markdown / filler."""
        start_idx = raw_response.find("{")
        end_idx = raw_response.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"No valid JSON structure found in output: {raw_response}")
        return json.loads(raw_response[start_idx : end_idx + 1])

    # --------------------------------------
    # Retrieval
    # --------------------------------------
    def search_facility_vectors(self, patient_need: str, radius_km: int) -> list[dict]:
        """Query the vector index for facilities matching the clinical need."""
        print("\n[System] Embedding query and searching vector index...")

        query_vector = self.embedding_model.encode(patient_need).tolist()

        index = self.vs_client.get_index(
            endpoint_name=self.vector_search_endpoint,
            index_name=self.index_name,
        )

        results = index.similarity_search(
            query_vector=query_vector,
            columns=["name", "combined_text"],
            num_results=TOP_K_RESULTS,
        )

        formatted_results: list[dict] = []
        for row in results.get("result", {}).get("data_array", []):
            formatted_results.append({"name": row[0], "combined_text": row[1]})
        return formatted_results

    # --------------------------------------
    # Agent 1 - Capability ranking
    # --------------------------------------
    def rank_regional_facilities(self, user_zip: str, patient_need: str) -> dict:
        """Filter by radius and rank based on clinical dependency verification.

        Returns a payload shaped to match the UI's `ranked_facilities` contract
        (see `ui/dummy_api_response_v2.json`):

            {
              "request_id": "...",
              "metadata": {...},
              "ranked_facilities": [
                {
                  "rank": 1,
                  "facility": {"name": "...", "type": "...", "specialties": [...]},
                  "scores": {"composite_0_100": 95},
                  "reasons": [...],
                  "raw_text": "..."   # combined_text from vector search
                },
                ...
              ]
            }
        """
        regional_locations = self.search_facility_vectors(patient_need, SEARCH_RADIUS_KM)
        if not regional_locations:
            return {
                "request_id": f"safemd_{user_zip}",
                "metadata": {"patient_need": patient_need, "user_zip": user_zip},
                "ranked_facilities": [],
            }

        context_corpus = "\n".join(
            f"Clinic {loc['name']}: {loc['combined_text']}" for loc in regional_locations
        )

        system_prompt = (
            "You are a Clinical Dependency Validator. "
            f"Review the regional facilities and rank them based on their ability to treat: '{patient_need}'. "
            "1. Verify claims against dependencies (e.g., a broken leg needs an X-ray AND an orthopedic specialist). "
            f"2. Assign a trust score ({TRUST_SCORE_MIN}-{TRUST_SCORE_MAX}). "
            "3. Output a JSON object with the following exact schema:\n"
            "{\n"
            '  "ranked_facilities": [\n'
            "    {\n"
            '      "name": "<facility name from context>",\n'
            '      "trust_score": <int 0-100>,\n'
            '      "specialty": "<best-matching specialty for the patient need>",\n'
            '      "reasons": ["<short reason 1>", "<short reason 2>"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Sort the list by highest trust_score first. Use ONLY facilities that appear in the context."
        )

        try:
            response = self._client.predict(
                endpoint=self.ranking_model,
                inputs={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context_corpus}"},
                    ],
                    "temperature": DEFAULT_TEMPERATURE,
                    "max_tokens": MAX_TOKENS_RANKING,
                },
            )
            raw_content = response["choices"][0]["message"]["content"]
            llm_output = self._parse_llm_json(raw_content)
        except Exception as e:
            print(f"[Error] Ranking model failed: {e}")
            return {
                "error": str(e),
                "request_id": f"safemd_{user_zip}",
                "metadata": {"patient_need": patient_need, "user_zip": user_zip},
                "ranked_facilities": [],
            }

        raw_text_by_name: dict[str, str] = {
            loc["name"]: loc["combined_text"] for loc in regional_locations
        }

        normalized: list[dict] = []
        for idx, item in enumerate(llm_output.get("ranked_facilities", []), start=1):
            name = item.get("name") or ""
            trust = item.get("trust_score")
            specialty = item.get("specialty") or ""
            reasons = item.get("reasons") or []
            specialties = [s.strip() for s in str(specialty).split(",") if s.strip()]
            normalized.append(
                {
                    "rank": idx,
                    "facility": {
                        "name": name,
                        "type": None,
                        "specialties": specialties or [str(specialty)] if specialty else [],
                    },
                    "scores": {"composite_0_100": trust},
                    "reasons": reasons,
                    "raw_text": raw_text_by_name.get(name, ""),
                }
            )

        return {
            "request_id": f"safemd_{user_zip}",
            "metadata": {
                "patient_need": patient_need,
                "user_zip": user_zip,
                "model": self.ranking_model,
                "top_k": TOP_K_RESULTS,
                "vector_index": self.index_name,
            },
            "ranked_facilities": normalized,
        }

    # --------------------------------------
    # Agent 2 - Triage
    # --------------------------------------
    def triage_incident(self, raw_incident: str) -> dict:
        """Extract structured metadata from a raw free-text incident."""
        system_prompt = (
            "You are a Triage Agent. Extract the following from the raw incident report: "
            "'Department', 'Severity_Level', 'Equipment_Involved', and 'Standardized_Summary'. "
            "Output strictly as a JSON object."
        )

        try:
            response = self._client.predict(
                endpoint=self.triage_model,
                inputs={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": raw_incident},
                    ],
                    "temperature": DEFAULT_TEMPERATURE,
                    "max_tokens": MAX_TOKENS_TRIAGE,
                },
            )
            raw_content = response["choices"][0]["message"]["content"]
            return self._parse_llm_json(raw_content)
        except Exception as e:
            print(f"[Error] Triage model failed: {e}")
            return {"error": "Triage failed", "raw_text": raw_incident[:50]}

    # --------------------------------------
    # Agent 3 - DMAIC / A3 report
    # --------------------------------------
    def generate_dmaic_analysis(self, location_id: str, raw_incidents: list[dict]) -> str:
        """Generate an A3 report for a specific facility."""
        if not raw_incidents:
            return "No incident reports found for analysis."

        print(f"\n[System] Triaging {len(raw_incidents)} incidents for {location_id}...")
        structured_incidents = [self.triage_incident(json.dumps(inc)) for inc in raw_incidents]
        report_corpus = json.dumps(structured_incidents, indent=2)

        system_prompt = (
            f"You are a Quality Improvement (QI) AI analyzing data exclusively for Facility ID: {location_id}. "
            "Survey the provided triaged incidents and apply the DMAIC (Define, Measure, Analyze, "
            "Improve, and Control) framework. "
            "Connect the dots to identify hidden systemic patterns. "
            "Output an actionable, Lean-compliant A3 report detailing root-cause analysis and prioritized "
            "action points tailored to this specific location."
        )

        try:
            response = self._client.predict(
                endpoint=self.reporting_model,
                inputs={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Triaged Incidents:\n{report_corpus}"},
                    ],
                    "temperature": DEFAULT_TEMPERATURE,
                    "max_tokens": MAX_TOKENS_REPORTING,
                },
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error] Reporting model failed: {e}"


# ==========================================
# 4. Helpers
# ==========================================
def load_incidents(filepath: str) -> list[dict]:
    """Load incidents from JSON or JSONL.  Returns mock data if the file is missing."""
    if not os.path.exists(filepath):
        print(f"[Warning] File {filepath} not found. Using mock data for testing.")
        return [
            {
                "location_id": "Aatmiy Critical Care Hospital",
                "text": "Patient transported to radiology, but elevator 3 was broken, causing a 20 minute delay.",
            },
            {
                "location_id": "Aatmiy Critical Care Hospital",
                "text": "Orthopedic casting supplies running low in Ward B. Almost ran out during night shift.",
            },
        ]

    incidents: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            incidents = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    incidents.append(json.loads(line))
    return incidents
