"""Log the Week 3 imbalance-study runs and precision-recall curves to MLflow."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.baselines.logistic_regression_baseline import (
    load_dataset as load_baseline_dataset,
    save_model as save_baseline_model,
    train_baseline_model,
)
from src.baselines.logistic_regression_class_weight import (
    save_model as save_class_weight_model,
    train_class_weight_model,
)
from src.baselines.logistic_regression_focal_loss import (
    load_dataset as load_focal_dataset,
    save_model as save_focal_model,
    train_focal_loss_model,
)
from src.baselines.logistic_regression_smote import (
    save_model as save_smote_model,
    train_smote_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "states_socal_1787874146_pairs_labeled.csv"
)
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()
DEFAULT_EXPERIMENT_NAME = "week3_imbalance_study"


# This function writes the PR-curve arrays from the shared evaluation harness to a
# CSV artifact so every MLflow run keeps the exact threshold data behind its plot.
def write_pr_curve_csv(pr_curve: dict[str, np.ndarray], output_path: Path) -> None:
    thresholds = np.asarray(pr_curve["thresholds"])
    precision = np.asarray(pr_curve["precision"])
    recall = np.asarray(pr_curve["recall"])

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["precision", "recall", "threshold"])
        for index in range(len(precision)):
            threshold = thresholds[index] if index < len(thresholds) else ""
            writer.writerow([precision[index], recall[index], threshold])


# This function creates the precision-recall visualization used as an MLflow
# artifact for one imbalance strategy.
def write_pr_curve_plot(
    pr_curve: dict[str, np.ndarray],
    average_precision: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        pr_curve["recall"],
        pr_curve["precision"],
        label=f"Average precision = {average_precision:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Week 3 precision-recall curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# This function records one model's parameters, metrics, model artifact, and PR
# curve artifacts in a single MLflow run.
def log_run(
    strategy: str,
    model: object,
    metrics: dict,
    save_model_function: object,
    output_path: Path,
    artifact_directory: Path,
) -> None:
    mlflow.start_run(run_name=strategy)
    try:
        mlflow.set_tags({"week": "3", "study": "imbalance", "strategy": strategy})
        mlflow.log_params({"strategy": strategy, "threshold": metrics["threshold"]})

        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and key != "threshold"
        }
        mlflow.log_metrics({key: float(value) for key, value in scalar_metrics.items()})

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_model_function(model, output_path)
        mlflow.log_artifact(str(output_path), artifact_path="model")

        curve_csv = artifact_directory / f"{strategy}_precision_recall_curve.csv"
        curve_plot = artifact_directory / f"{strategy}_precision_recall_curve.png"
        write_pr_curve_csv(metrics["pr_curve"], curve_csv)
        write_pr_curve_plot(metrics["pr_curve"], metrics["average_precision"], curve_plot)
        mlflow.log_artifact(str(curve_csv), artifact_path="precision_recall_curve")
        mlflow.log_artifact(str(curve_plot), artifact_path="precision_recall_curve")
    finally:
        mlflow.end_run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI. Defaults to the local mlruns folder.",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    args = parser.parse_args()

    if args.tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X, y = load_baseline_dataset(args.data_path)
    X_focal, y_focal = load_focal_dataset(args.data_path)
    if len(X) != len(X_focal) or not np.array_equal(y.to_numpy(), y_focal):
        raise ValueError("Baseline and focal-loss loaders produced different targets.")

    with TemporaryDirectory() as temporary_directory:
        artifact_directory = Path(temporary_directory)

        baseline_model, baseline_metrics, _ = train_baseline_model(X, y)
        log_run(
            "baseline",
            baseline_model,
            baseline_metrics,
            save_baseline_model,
            PROJECT_ROOT / "models" / "week3_logistic_regression_baseline.joblib",
            artifact_directory,
        )

        smote_model, smote_metrics, _ = train_smote_model(X, y)
        log_run(
            "smote",
            smote_model,
            smote_metrics,
            save_smote_model,
            PROJECT_ROOT / "models" / "week3_logistic_regression_smote.joblib",
            artifact_directory,
        )

        class_weight_model, class_weight_metrics, _ = train_class_weight_model(X, y)
        log_run(
            "class_weight",
            class_weight_model,
            class_weight_metrics,
            save_class_weight_model,
            PROJECT_ROOT / "models" / "week3_logistic_regression_class_weight.joblib",
            artifact_directory,
        )

        focal_model, focal_metrics = train_focal_loss_model(X_focal, y_focal)
        log_run(
            "focal_loss",
            focal_model,
            focal_metrics,
            save_focal_model,
            PROJECT_ROOT / "models" / "week3_logistic_regression_focal_loss.pt",
            artifact_directory,
        )

    print(f"MLflow experiment: {args.experiment_name}")
    print(f"Tracking URI: {args.tracking_uri}")
    print("Logged runs: baseline, smote, class_weight, focal_loss")


if __name__ == "__main__":
    main()