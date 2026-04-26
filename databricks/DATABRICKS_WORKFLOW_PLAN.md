# Databricks AI Gateway Workflow Plan

This plan operationalizes your Step A-E design for a hospital-facility verification and equity intelligence system.

It is aligned to the SafeMD.ai judging rubric in `AI Hackathon (1).pdf`:
- Discovery and Verification (35%)
- Intelligent Document Parsing (30%)
- Social Impact and Utility (25%)
- User Experience and Transparency (10%)

## 1) Architecture (Databricks-native)

- **Ingestion layer**: CSV/Delta ingestion into Bronze table.
- **Triage agent**: Cleansing + intelligent text consolidation + vector retrieval.
- **Reasoning agent**: Claim extraction from facility context.
- **Validator agent**: Evidence cross-check + trust score.
- **IDP/schema layer**: Strict mapping to Virtue Foundation schema.
- **Reporting agent**: DMAIC + A3 summaries for administrators.
- **Equity agent**: Geospatial desert/risk tagging via UC + census tables.
- **Serving layer**: API payload for frontend map/dashboard + correction loop.

## 2) API choices (build now vs. scale later)

### Build stage (free APIs)
- Route Databricks AI Gateway to a free-tier model provider (for example Groq free model routes) for:
  - `PRIMARY_LLM_ENDPOINT`
  - `VALIDATOR_LLM_ENDPOINT`
  - `REPORTING_LLM_ENDPOINT`

This keeps cost low while validating orchestration, schema quality, and UX.

### Production recommendation (best quality)
- Replace primary and validator with **Anthropic Claude Sonnet** via Databricks External Model endpoint.
- Keep reporting on DBRX/Llama family if cost-sensitive, or upgrade to the same Claude endpoint for consistency.

## 3) Step-by-step implementation map

## Step A - Intake, cleansing, intelligent parsing (Triage)
- Replace nulls with explicit `UNKNOWN`.
- Concatenate `description` + sparse fields (`specialties`, `procedure`, `equipment`, `capability`) into `triage_text`.
- Embed/index in Mosaic AI Vector Search.
- Query top-k similar records for contextual grounding.

## Step B - Extraction + self-correction loop (Reasoning + Validator)
- Primary agent extracts claims and evidence snippets.
- Validator agent checks each claim against the same retrieval context.
- Validator explicitly distinguishes **Resident Specialist** vs **Visiting Consultant** signals.
- Validator runs anomaly checks for “high trust but vague details.”
- Emit:
  - `SUPPORTED | PARTIAL | UNSUPPORTED` verdict per claim
  - `trust_score` 0-100
  - contradiction count + human-review boolean
  - anomaly flags + reasoning trace summary for UI transparency

## Step C - Schema alignment (IDP)
- Map validated output to Virtue Foundation schema fields.
- Separate:
  - `confirmed_capabilities`
  - `unverified_claims`
  - `missing_fields`
- Preserve source snippets for traceability.

## Step D - Quality improvement / DMAIC (Reporting)
- Aggregate validated records by district/state/operator.
- Generate:
  - Define/Measure/Analyze/Improve/Control JSON blocks
  - A3-ready report payloads
  - risk trends over specialties/capability gaps

## Step E - Policy + geospatial sync (Equity)
- Use a dynamic `GOLDEN_HOUR_RADIUS_KM` config variable.
- Cross-reference facility readiness with census/health-demand Delta tables.
- Produce district/PIN “medical desert” rankings and service-gap tags.
- Add explicit filters for high-acuity gaps: Oncology, Dialysis, Trauma.

## 4) UI transformation contract (for your next phase)

Every facility card should receive:
- `trust_score`
- `needs_human_review`
- `anomaly_flags`
- `reasoning_trace`
- `confirmed_capabilities`
- `unverified_claims`
- `missing_fields`
- `staffing_signals` (resident vs visiting)
- `evidence[]` with exact snippets
- `urgent_need`
- `one_resource_away`
- `ui_actions.allow_correction_submit`

This supports:
- source-citation modal on click
- trust/confidence filters
- “urgent need” and “one-resource-away” planner filters
- “Correct this entry” feedback loop into retraining queue
- executive district dashboard for A3 + DMAIC at-a-glance

## 5) Databricks Job orchestration

Suggested daily job graph:
1. Bronze ingest -> cleanse
2. Vector index update
3. Agent pipeline execution
4. Schema + quality table upsert
5. DMAIC report generation
6. API serving table refresh
7. Feedback corrections merge

## 6) Minimal tables to maintain

- `bronze_facilities_raw`
- `silver_facilities_cleaned`
- `silver_facilities_validated`
- `gold_facility_trust_and_gaps`
- `gold_dmaic_district_reports`
- `feedback_facility_corrections`

## 7) File you can run now

- `databricks/ai_gateway_workflow.py`

It is implementation-oriented and designed for Databricks serving endpoints + AI Gateway routes.

## 8) Production API recommendation

- **Best quality (recommended): Anthropic Claude Sonnet via Databricks External Model endpoint**
  - Why: strongest reliability for contradiction detection, evidence-grounded extraction, and nuanced staffing/resource interpretation.
  - Use for: Triage/Reasoning + Validator + Schema mapping.
- **Cost-aware hybrid:** keep DMAIC narrative on DBRX/Llama if needed.

