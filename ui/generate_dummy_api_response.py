import json
from pathlib import Path

import numpy as np


def main() -> None:
    out_path = Path("ui/dummy_api_response.json")
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
    rng = np.random.default_rng(42)
    rows = []
    for i in range(200):
        ftype = facility_types[i % len(facility_types)]
        rows.append(
            {
                "rank": i + 1,
                "place_name": f"SafeMD Ranked Facility {i+1:03d}",
                "specialty": specialties[i % len(specialties)],
                "score": float(np.round(rng.uniform(35, 99), 1)),
                "facility_type_id": ftype,
            }
        )
    payload = {"results": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} results to {out_path}")


if __name__ == "__main__":
    main()
