"""Train the Week 3 logistic regression baseline with focal loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluation.harness import build_evaluation_harness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "states_socal_1787874146_pairs_labeled.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "week3_logistic_regression_focal_loss.pt"

FEATURE_COLUMNS = [
    "lateral_distance_nm",
    "closing_speed_mps",
    "bearing_difference_deg",
    "vertical_separation_m",
    "time_to_cpa_s",
]


# This function loads the same cleaned feature matrix used by the earlier Week 3
# baselines so the only difference is the imbalance-handling objective. Non-finite
# values are removed before training because focal loss still requires valid numeric input.
def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    required_columns = [*FEATURE_COLUMNS, "collision_risk"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    clean_df = df[required_columns].copy()
    for column in FEATURE_COLUMNS:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")
    clean_df = clean_df.replace([float("inf"), float("-inf")], float("nan")).dropna().copy()
    X = clean_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = clean_df["collision_risk"].astype(int).to_numpy(dtype=int)
    return X, y


# This function defines the focal loss used for the rare positive class. The model
# is asked to pay more attention to hard, minority-class examples instead of simply
# optimizing ordinary cross-entropy on the majority class.
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating_factor = (1.0 - p_t).pow(gamma)
    return (alpha_factor * modulating_factor * ce_loss).mean()


# This function trains a single logistic model in PyTorch with the focal loss
# objective. This matches the Week 3 requirement to compare a focal-loss version
# against the plain and SMOTE baselines using the same evaluation harness.
def train_focal_loss_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    learning_rate: float = 0.01,
    epochs: int = 400,
) -> tuple[torch.nn.Module, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train_scaled.shape[1], 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = focal_loss(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        y_prob = torch.sigmoid(test_logits).numpy().ravel()

    metrics = build_evaluation_harness(y_test, y_prob, threshold=0.5)
    return model, metrics


# This function saves the trained model so it can be compared against the plain and
# SMOTE baselines and later logged to MLflow with the Week 3 experiment results.
def save_model(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


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
        help="Where to save the trained focal-loss model.",
    )
    args = parser.parse_args()

    X, y = load_dataset(args.data_path)
    model, metrics = train_focal_loss_model(X, y)
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
