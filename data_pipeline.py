import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_CANDIDATES = [
    "engagement_metrics_n_engagements",
    "engagement_metrics_n_likes",
    "engagement_metrics_n_followers",
    "capacity",
    "numberDoctors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end preprocessing + feature importance pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--target",
        default=None,
        help="Target column to model. If omitted, a known candidate will be selected.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Modeling objective. auto infers from target distribution.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for processed data, model outputs, and reports.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Holdout fraction for evaluation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _to_list(value: Any) -> list[Any]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                loaded = ast.literal_eval(text)
                if isinstance(loaded, list):
                    return loaded
            except (ValueError, SyntaxError):
                return []
    return []


def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_bool(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return np.nan


def infer_target(df: pd.DataFrame, explicit_target: str | None) -> str:
    if explicit_target:
        if explicit_target not in df.columns:
            raise ValueError(f"Target '{explicit_target}' not found in CSV columns.")
        return explicit_target
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No target supplied and no default candidate present. "
        "Pass --target explicitly."
    )


def infer_task(y: pd.Series, user_task: str) -> str:
    if user_task != "auto":
        return user_task
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().mean() > 0.95 and y_num.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    list_like_cols = [
        "phone_numbers",
        "websites",
        "affiliationTypeIds",
        "specialties",
        "procedure",
        "equipment",
        "capability",
    ]
    for col in list_like_cols:
        if col in data.columns:
            parsed = data[col].apply(_to_list)
            data[f"{col}_count"] = parsed.apply(len)

    bool_cols = [
        "affiliated_staff_presence",
        "custom_logo_presence",
    ]
    for col in bool_cols:
        if col in data.columns:
            data[col] = data[col].apply(_safe_bool)

    if "description" in data.columns:
        desc = data["description"].fillna("").astype(str)
        data["description_char_len"] = desc.str.len()
        data["description_word_len"] = desc.str.split().str.len()

    if "name" in data.columns:
        name = data["name"].fillna("").astype(str)
        data["name_char_len"] = name.str.len()

    numeric_cols = [
        "numberDoctors",
        "capacity",
        "distinct_social_media_presence_count",
        "number_of_facts_about_the_organization",
        "post_metrics_post_count",
        "engagement_metrics_n_followers",
        "engagement_metrics_n_likes",
        "engagement_metrics_n_engagements",
        "latitude",
        "longitude",
        "yearEstablished",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = _safe_num(data[col])

    if "post_metrics_most_recent_social_media_post_date" in data.columns:
        dates = pd.to_datetime(
            data["post_metrics_most_recent_social_media_post_date"], errors="coerce"
        )
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        data["days_since_last_social_post"] = (now - dates).dt.days

    drop_raw_columns = [
        "phone_numbers",
        "websites",
        "description",
        "procedure",
        "equipment",
        "capability",
        "specialties",
    ]
    existing_drop = [col for col in drop_raw_columns if col in data.columns]
    data = data.drop(columns=existing_drop)

    return data


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols and X[c].dtype == object
    ]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    target = infer_target(df, args.target)

    engineered = engineer_features(df)
    if target not in engineered.columns:
        engineered[target] = _safe_num(df[target])

    engineered = engineered.dropna(subset=[target]).copy()
    y = engineered[target]
    X = engineered.drop(columns=[target])

    task = infer_task(y, args.task)
    if task == "classification":
        y = y.fillna("unknown").astype(str)
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        stratify = y if y.nunique() <= 30 else None
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_mask = y.notna()
        X = X.loc[valid_mask].copy()
        y = y.loc[valid_mask]
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=args.seed,
            n_jobs=-1,
        )
        stratify = None

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    engineered.to_csv(output_dir / "processed_data.csv", index=False)
    fi_df.to_csv(output_dir / "feature_importances.csv", index=False)
    joblib.dump(pipeline, output_dir / "trained_pipeline.joblib")

    report = {
        "input_file": str(input_path),
        "output_dir": str(output_dir.resolve()),
        "target": target,
        "task": task,
        "rows_total": int(len(df)),
        "rows_after_target_filter": int(len(engineered)),
        "n_features_pre_encoding": int(X.shape[1]),
        "n_numeric_features": int(len(numeric_cols)),
        "n_categorical_features": int(len(categorical_cols)),
        "metrics": metrics,
        "top_20_features": fi_df.head(20).to_dict(orient="records"),
    }
    (output_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2))

    print("Pipeline complete.")
    print(f"Target: {target}")
    print(f"Task: {task}")
    print(f"Metrics: {metrics}")
    print(f"Artifacts written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
