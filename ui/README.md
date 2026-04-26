# SafeMD.ai UI (Local)

## 1) Generate local test data (no Databricks required)

From project root:

```powershell
python databricks/ai_gateway_workflow.py "VF_Hackathon_Dataset_India_Large.xlsx.csv" --mode local --max-records 300 --output-path databricks_agent_output.json
```

## 2) Run the UI

```powershell
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

## 2.5) Generate synthetic incident data for live-map testing

```powershell
python ui/generate_incident_reports.py --count 3000 --out-jsonl ui/incidents.jsonl --out-analytics ui/incident_analytics.json
```

This creates:
- `ui/incidents.jsonl` (3,000 incident reports)
- `ui/incident_analytics.json` (aggregate incident analysis summary)

## 3) What this UI includes

- Trust/confidence filtering
- Urgent need + one-resource-away views
- ICU support-gap filter
- Facility map
- Evidence snippet table for source citation
- Reasoning trace visibility
- Executive DMAIC dashboard
- "Correct this entry" feedback capture (local JSONL queue)
- Separate tabs for live map, incident reporting, incident analysis, and executive dashboard
- Map marker color updates for active incidents (power outage highlighted in red)

## 4) Move from local to Databricks mode

When ready for real endpoint calls:

```powershell
$env:DATABRICKS_HOST="https://<workspace-url>"
$env:DATABRICKS_TOKEN="<token>"
python databricks/ai_gateway_workflow.py "VF_Hackathon_Dataset_India_Large.xlsx.csv" --mode databricks --max-records 1000 --output-path databricks_agent_output.json
```
