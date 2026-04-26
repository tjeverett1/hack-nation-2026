"""
SafeMD.ai - end-to-end test runner.

Mirrors the `if __name__ == "__main__":` block from the original Databricks
notebook, but works from a laptop / CI runner.

Run:
    .venv/bin/python -m pipeline.run_pipeline                # all three agents
    .venv/bin/python -m pipeline.run_pipeline --only ranking # one agent
    .venv/bin/python -m pipeline.run_pipeline --only triage
    .venv/bin/python -m pipeline.run_pipeline --only dmaic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE we import the pipeline (the pipeline checks env on import-time
# of its constructor, so .env must be in place first).
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from pipeline.safemd_pipeline import SafeMDPipeline, load_incidents  # noqa: E402


# ==========================================
# Test fixtures
# ==========================================
TARGET_ZIP_CODE = "380009"
PATIENT_NEED = "broken leg requiring emergency orthopedic care and X-ray"
TARGET_HOSPITAL_ID = "Aatmiy Critical Care Hospital"

# Incidents file - default points to the workspace path you used in the
# notebook.  When running locally, drop a copy at this location or pass
# --incidents-path.
DEFAULT_INCIDENTS_PATH = "/Workspace/Users/tjeveret@mit.edu/incidents.json"

SAMPLE_INCIDENT = (
    "Patient in Ward 4 experienced a delay in receiving pain medication due to a "
    "Pyxis machine malfunction. Nurse had to override manually. High priority as "
    "patient was in severe distress."
)


# ==========================================
# Individual test cases
# ==========================================
def test_ranking(pipeline: SafeMDPipeline) -> None:
    print("=== TESTING CAPABILITY RANKING ===")
    print(f"Patient Need: {PATIENT_NEED}")
    ranking_results = pipeline.rank_regional_facilities(TARGET_ZIP_CODE, PATIENT_NEED)
    print(json.dumps(ranking_results, indent=2))


def test_triage(pipeline: SafeMDPipeline) -> None:
    print("\n=== TESTING INCIDENT TRIAGE ===")
    print(f"Raw Incident: '{SAMPLE_INCIDENT}'")
    triage_result = pipeline.triage_incident(SAMPLE_INCIDENT)
    print("\nFull Extracted Metadata:")
    print(json.dumps(triage_result, indent=2))
    print(f"\n-> Isolated Severity Level: {triage_result.get('Severity_Level', 'Not found')}")


def test_dmaic(pipeline: SafeMDPipeline, incidents_path: str) -> None:
    print("\n=== TESTING DMAIC GENERATION ===")
    print(f"Target Hospital: {TARGET_HOSPITAL_ID}")

    loaded_data = load_incidents(incidents_path)
    target_hospital_incidents = [
        inc for inc in loaded_data if inc.get("location_id") == TARGET_HOSPITAL_ID
    ]

    dmaic_report = pipeline.generate_dmaic_analysis(TARGET_HOSPITAL_ID, target_hospital_incidents)
    print("\n--- GENERATED A3 REPORT ---")
    print(dmaic_report)


# ==========================================
# CLI
# ==========================================
def main() -> int:
    parser = argparse.ArgumentParser(description="SafeMD.ai pipeline test runner")
    parser.add_argument(
        "--only",
        choices=["ranking", "triage", "dmaic", "all"],
        default="all",
        help="Run a single agent's test only",
    )
    parser.add_argument(
        "--incidents-path",
        default=DEFAULT_INCIDENTS_PATH,
        help="Path to JSON / JSONL incidents file (mock data used if missing)",
    )
    args = parser.parse_args()

    pipeline = SafeMDPipeline()

    if args.only in ("ranking", "all"):
        test_ranking(pipeline)
    if args.only in ("triage", "all"):
        test_triage(pipeline)
    if args.only in ("dmaic", "all"):
        test_dmaic(pipeline, args.incidents_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
