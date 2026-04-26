import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    out_path = Path("ui/dummy_api_response_v2.json")
    facility_types = ["clinic", "dentist", "hospital", "healthCenter", "traumaCenter"]
    specialties = [
        "Primary Care",
        "Emergency",
        "Optometry",
        "Sports Medicine",
        "Dental",
        "Oncology",
        "Dialysis",
        "Trauma",
        "Neonatal",
    ]
    reasons = [
        "High trust score and strong consistency signals",
        "Low active incident burden",
        "Close proximity to user postal zone",
        "Capability coverage matches requested condition",
        "Good staffing stability indicators",
    ]
    rng = np.random.default_rng(7)
    ranked = []
    for i in range(200):
        ftype = facility_types[i % len(facility_types)]
        spec1 = specialties[i % len(specialties)]
        spec2 = specialties[(i + 3) % len(specialties)]
        ranked.append(
            {
                "rank": i + 1,
                "facility": {
                    "id": f"fac_{i+1:04d}",
                    "name": f"SafeMD Future API Facility {i+1:03d}",
                    "type": ftype,
                    "specialties": [spec1, spec2],
                },
                "scores": {
                    "score": float(np.round(rng.uniform(30, 99), 1)),
                    "trust_0_100": float(np.round(rng.uniform(35, 99), 1)),
                    "safety_0_100": float(np.round(rng.uniform(30, 99), 1)),
                },
                "distance_km": float(np.round(rng.uniform(0.8, 80), 1)),
                "incident_open_count": int(rng.integers(0, 12)),
                "reasons": [reasons[i % len(reasons)], reasons[(i + 2) % len(reasons)]],
            }
        )

    payload = {
        "request_id": "dummy_req_0001",
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": "dummy-v2",
            "top_k": 200,
            "notes": "Future contract mock for API integration.",
        },
        "ranked_facilities": ranked,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote future contract with {len(ranked)} rows to {out_path}")


if __name__ == "__main__":
    main()
