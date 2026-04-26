import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def load_facilities(path: Path) -> list[dict]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("facility_results", [])
        if rows:
            return rows
    # Fallback from CSV if pipeline output doesn't exist.
    csv_path = Path("VF_Hackathon_Dataset_India_Large.xlsx.csv")
    if not csv_path.exists():
        raise FileNotFoundError("No facility source found. Generate databricks_agent_output.json or place CSV in root.")
    df = pd.read_csv(csv_path, low_memory=False).head(2000)
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "facility_id": int(idx),
                "facility_name": str(row.get("name", "UNKNOWN")),
                "city": str(row.get("address_city", "UNKNOWN")),
                "state": str(row.get("address_stateOrRegion", "UNKNOWN")),
                "pincode": str(row.get("address_zipOrPostcode", "UNKNOWN")),
            }
        )
    return rows


def synthesize_incidents(facilities: list[dict], n: int, seed: int) -> list[dict]:
    random.seed(seed)
    scopes = ["Single Patient", "Single Department", "Whole Hospital", "Regional Network", "Supply Chain", "Staff Safety"]
    types = [
        "Unexpected Outcome",
        "Insufficient Resources",
        "Insufficient Staff",
        "Equipment Failure",
        "Infrastructure Failure",
        "Near Miss",
    ]
    subtype_by_type = {
        "Unexpected Outcome": ["Post-op complication", "Unexpected transfer", "Readmission spike"],
        "Insufficient Resources": ["Oxygen Shortage", "Blood Product Delay", "Dialysis Consumables Low"],
        "Insufficient Staff": ["No Anesthesiologist", "No Resident Specialist", "Nurse Shortage"],
        "Equipment Failure": ["Ventilator Failure", "Dialysis Unit Down", "Imaging Downtime"],
        "Infrastructure Failure": ["Power Outage", "Water Supply Failure", "Network Downtime"],
        "Near Miss": ["Medication Error", "Patient ID Mismatch", "Specimen Label Issue"],
    }
    severities = ["low", "medium", "high", "critical"]
    status_weights = [0.55, 0.25, 0.20]  # open, monitoring, resolved
    severity_weights = [0.25, 0.40, 0.25, 0.10]
    descriptions = [
    "Team reported a disruption affecting standard care workflow.",
    "Frontline staff flagged resource constraints impacting timely treatment.",
    "Incident suggests temporary loss of functional capacity in one or more units.",
    "Escalation requested to district admin for rapid mitigation support.",
    "Critical shortage of single-use sterile consumables reported in the main surgical theater.",
    "Supply chain logistics failure resulted in the delivery of expired reagents for the laboratory.",
    "Oxygen pressure fluctuations detected; switching to backup manifold as a precautionary measure.",
    "Blood bank reserves for rare types reached critical threshold; elective procedures postponed.",
    "Cold chain integrity for vaccines compromised during last-mile transit; investigation initiated.",
    "Personal Protective Equipment (PPE) inventory burn rate exceeded weekly forecast by 40%.",
    "Essential medication stock-out reported; procurement team exploring alternative local vendors.",
    "Double-shift fatigue cited as a contributing factor in a documented administrative oversight.",
    "Unscheduled absence of senior technician resulted in the closure of the imaging wing.",
    "Security intervention required following a verbal escalation in the emergency department waiting area.",
    "Staffing ratios fell below mandated levels following a localized flu outbreak among personnel.",
    "Inter-shift handover incomplete; vital information regarding patient allergies was nearly missed.",
    "Temporary specialist hire required to cover specialized pediatric oncology vacancy.",
    "HVAC system failure in the sterile storage zone; inventory being relocated to climate-controlled pods.",
    "Hospital Information System (HIS) experiencing intermittent database locks, slowing data entry.",
    "Primary internet gateway down; staff operating on manual paper-backup procedures.",
    "Water filtration system at the dialysis center triggered a high-conductivity alarm.",
    "Elevator malfunction affecting the transport of critical care patients between floors.",
    "Backup generator failed its weekly load test; facility at risk during municipal power swings.",
    "Fire suppression system triggered in a non-clinical storage area; no patient impact reported.",
    "Regional network congestion preventing the upload of high-resolution radiology files.",
    "External disaster in the district has triggered the facility's mass-casualty surge protocol.",
    "Patient vitals stabilized after initial equipment discrepancy; monitoring for secondary complications.",
    "Clinical lead noted a significant deviation from standard operating procedure during triage.",
    "Post-surgical recovery delayed due to unavailability of specialized monitoring hardware.",
    "Anomalous reading on diagnostic equipment prompted a full manual reassessment of the patient.",
    "Cross-departmental transfer delayed, resulting in a temporary breach of stabilization window.",
    "Medication administration timing disrupted by pharmacy communication system lag.",
    "Discharge planning halted due to lack of local community support for high-risk patients.",
    "Urgent pathology results delayed; clinical team proceeded with empirical treatment protocols."
]

    incidents = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        facility = random.choice(facilities)
        incident_type = random.choice(types)
        subtype = random.choice(subtype_by_type[incident_type])
        ts = now - timedelta(hours=random.randint(1, 24 * 45))
        status = random.choices(["open", "monitoring", "resolved"], weights=status_weights, k=1)[0]
        severity = random.choices(severities, weights=severity_weights, k=1)[0]
        incidents.append(
            {
                "incident_id": f"inc_seed_{i+1}",
                "timestamp_utc": ts.isoformat(),
                "facility_id": int(facility["facility_id"]),
                "facility_name": facility["facility_name"],
                "incident_scope": random.choice(scopes),
                "incident_type": incident_type,
                "incident_subtype": subtype,
                "description": f"{random.choice(descriptions)} ({subtype})",
                "severity": severity,
                "status": status,
            }
        )
    return incidents


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_analytics(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    summary = {
        "total_incidents": int(len(df)),
        "open_incidents": int((df["status"] == "open").sum()),
        "critical_incidents": int((df["severity"] == "critical").sum()),
        "top_incident_types": df["incident_type"].value_counts().head(10).to_dict(),
        "top_subtypes": df["incident_subtype"].value_counts().head(10).to_dict(),
        "facilities_with_open_power_outage": int(
            len(
                df[
                    (df["status"] == "open")
                    & (df["incident_subtype"] == "Power Outage")
                ]["facility_id"].unique()
            )
        ),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic incident reports for SafeMD.ai UI testing.")
    parser.add_argument("--count", type=int, default=3000, help="Number of incidents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--facility-json", default="databricks_agent_output.json", help="Facility JSON source")
    parser.add_argument("--out-jsonl", default="ui/incidents.jsonl", help="Incident output JSONL")
    parser.add_argument("--out-analytics", default="ui/incident_analytics.json", help="Incident analytics output JSON")
    args = parser.parse_args()

    facilities = load_facilities(Path(args.facility_json))
    incidents = synthesize_incidents(facilities, n=args.count, seed=args.seed)
    write_jsonl(Path(args.out_jsonl), incidents)
    build_analytics(incidents, Path(args.out_analytics))
    print(f"Generated {len(incidents)} incidents at {args.out_jsonl}")
    print(f"Analytics written to {args.out_analytics}")


if __name__ == "__main__":
    main()
