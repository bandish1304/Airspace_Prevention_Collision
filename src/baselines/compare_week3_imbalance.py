"""Compare Week 3 imbalance runs and select the strategy to carry forward."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()
DEFAULT_EXPERIMENT_NAME = "week3_imbalance_study"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "week3_imbalance_comparison.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "experiments" / "week3_imbalance_selection.md"
STRATEGIES = ("baseline", "smote", "class_weight", "focal_loss")
METRIC_COLUMNS = (
    "precision",
    "recall",
    "f1",
    "auroc",
    "average_precision",
    "false_negative_rate",
    "false_positive_rate",
)


# This function loads the latest MLflow run for each Week 3 strategy so the
# comparison is based on the most recent reproducible experiment outputs.
def load_latest_runs(tracking_uri: str, experiment_name: str) -> list[dict[str, float | str]]:
    if tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
    )
    latest_by_strategy: dict[str, dict] = {}
    for _, run in runs.iterrows():
        strategy = run.get("tags.strategy")
        if strategy in STRATEGIES and strategy not in latest_by_strategy:
            row: dict[str, float | str] = {
                "strategy": strategy,
                "run_id": str(run["run_id"]),
            }
            for metric in METRIC_COLUMNS:
                value = run.get(f"metrics.{metric}")
                if value is None:
                    raise ValueError(f"Missing metric {metric} for strategy {strategy}")
                row[metric] = float(value)
            latest_by_strategy[strategy] = row

    missing = [strategy for strategy in STRATEGIES if strategy not in latest_by_strategy]
    if missing:
        raise ValueError(f"Missing MLflow runs for strategies: {missing}")
    return [latest_by_strategy[strategy] for strategy in STRATEGIES]


# This function selects the safest strategy by minimizing missed risks first,
# then maximizing recall, F1, precision, average precision, and AUROC.
def select_strategy(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["false_negative_rate"]),
            -float(row["recall"]),
            -float(row["f1"]),
            -float(row["precision"]),
            -float(row["average_precision"]),
            -float(row["auroc"]),
        ),
    )[0]


# This function saves the exact metrics used for the Week 3 decision as a CSV
# artifact that can be included in the experiment results section.
def write_comparison_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["strategy", "run_id", *METRIC_COLUMNS]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# This function writes a short human-readable record of the selection rule and
# the winning strategy for later reporting and model handoff.
def write_selection_report(
    rows: list[dict[str, float | str]],
    winner: dict[str, float | str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Week 3 Imbalance Strategy Selection",
        "",
        "The selection prioritizes safety-critical detection in this order:",
        "false negative rate (lower), recall (higher), F1 (higher), precision (higher), average precision (higher), and AUROC (higher).",
        "",
        f"Selected strategy: **{winner['strategy']}**",
        "",
        "The selected strategy will carry forward to the Week 4 graph and later model experiments.",
        "",
        "| Strategy | Precision | Recall | F1 | AUROC | Average precision | False negative rate | False positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['auroc']):.4f} | "
            f"{float(row['average_precision']):.4f} | {float(row['false_negative_rate']):.4f} | "
            f"{float(row['false_positive_rate']):.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    rows = load_latest_runs(args.tracking_uri, args.experiment_name)
    winner = select_strategy(rows)
    write_comparison_csv(rows, args.output_path)
    write_selection_report(rows, winner, args.report_path)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="step7_selection"):
        mlflow.set_tags({"week": "3", "study": "imbalance", "step": "7"})
        mlflow.log_param("selected_strategy", winner["strategy"])
        mlflow.log_param("selection_priority", "fnr,recall,f1,precision,average_precision,auroc")
        mlflow.log_metrics(
            {
                f"selected_{metric}": float(winner[metric])
                for metric in METRIC_COLUMNS
            }
        )
        mlflow.log_artifact(str(args.output_path), artifact_path="comparison")
        mlflow.log_artifact(str(args.report_path), artifact_path="comparison")

    print(f"Selected strategy: {winner['strategy']}")
    print(f"Comparison table: {args.output_path}")
    print(f"Selection report: {args.report_path}")


if __name__ == "__main__":
    main()