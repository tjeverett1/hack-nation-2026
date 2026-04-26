# Data Processing Pipeline Usage

Run the full pipeline on the provided CSV:

```powershell
python data_pipeline.py --input "VF_Hackathon_Dataset_India_Large.xlsx - VF_Hackathon_Dataset_India_Larg.csv" --output-dir artifacts
```

If your project objective uses a different target column:

```powershell
python data_pipeline.py --input "VF_Hackathon_Dataset_India_Large.xlsx - VF_Hackathon_Dataset_India_Larg.csv" --target engagement_metrics_n_followers --task regression --output-dir artifacts
```

## What it does

1. Loads raw data and engineers robust numeric features from list/text/date fields.
2. Cleans booleans and numeric columns.
3. Builds a scikit-learn preprocessing + model pipeline.
4. Trains/evaluates a baseline model for your target objective.
5. Exports ranked feature importances for downstream AI-agent reasoning.

## Output artifacts

- `artifacts/processed_data.csv` - clean engineered dataset for AI-agent consumption.
- `artifacts/feature_importances.csv` - model-based feature ranking.
- `artifacts/trained_pipeline.joblib` - serialized end-to-end preprocessing + model pipeline.
- `artifacts/pipeline_report.json` - run metadata, evaluation metrics, and top features.

