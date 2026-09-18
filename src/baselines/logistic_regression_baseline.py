"""Train the Week 3 logistic regression baseline without imbalance handling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.harness import build_evaluation_harness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "states_socal_1787874146_pairs_labeled.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "week3_logistic_regression_baseline.joblib"

FEATURE_COLUMNS = [
    "lateral_distance_nm",
    "closing_speed_mps",
    "bearing_difference_deg",
    "vertical_separation_m",
    "time_to_cpa_s",
]


# This function loads the labeled pair dataset and keeps only the engineered pair
# features needed for the Week 3 baseline comparison. It excludes the target and
# metadata columns so the model learns from the actual collision-risk signals.
# Some engineered features, such as time_to_cpa_s, can be infinite when no future
# closest approach exists; those rows are dropped because scikit-learn cannot fit
# on non-finite inputs.
def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    required_columns = [*FEATURE_COLUMNS, "collision_risk"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for baseline training: {missing}")

    clean_df = df[required_columns].copy()
    for column in FEATURE_COLUMNS:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
    clean_df = clean_df.replace([float("inf"), float("-inf")], float("nan")).dropna().copy()
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["collision_risk"].astype(int)
    return X, y


# This function trains a standard logistic regression pipeline using a train/test
# split with stratification. It does not apply SMOTE, focal loss, or class weighting
# so it represents the plain baseline required by Week 3 step 2.
def train_baseline_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, dict, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = build_evaluation_harness(y_test.to_numpy(), y_prob, threshold=0.5)
    return pipeline, metrics, (X_train, X_test, y_train, y_test)


# This function saves the trained classifier to disk so it can be reused for later
# comparison runs and logged to MLflow in the next Week 3 step.
def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the labeled pair dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained baseline model.",
    )
    args = parser.parse_args()

    X, y = load_dataset(args.data_path)
    model, metrics, _ = train_baseline_model(X, y)
    save_model(model, args.output_path)

    print(f"Dataset rows: {len(X)}")
    print(f"Positive rate: {float(y.mean()):.4f}")
    print(f"Model saved to: {args.output_path}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"False negative rate: {metrics['false_negative_rate']:.4f}")


if __name__ == "__main__":
    main()
