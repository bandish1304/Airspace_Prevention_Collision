"""Label aircraft-pair features using configurable separation minima."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_LATERAL_SEPARATION_NM = 5.0
DEFAULT_VERTICAL_SEPARATION_FT = 1_000.0
METRES_PER_FOOT = 0.3048
REQUIRED_COLUMNS = {"lateral_distance_nm", "vertical_separation_m"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# Week 2, step 7: label pairs that breach both configured separation minima.
def label_collision_risk(
    pairs: pd.DataFrame,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
) -> pd.DataFrame:
    """Return pair features with a nullable binary ``collision_risk`` label.

    A pair is labeled 1 when its lateral and vertical separation are both at or
    below the configured limits; otherwise it is labeled 0. Rows missing either
    separation remain unlabelled (``<NA>``), rather than being assumed safe.
    """
    if lateral_separation_nm <= 0:
        raise ValueError("lateral_separation_nm must be greater than zero.")
    if vertical_separation_ft <= 0:
        raise ValueError("vertical_separation_ft must be greater than zero.")

    _validate_pair_columns(pairs)
    labeled_pairs = pairs.copy()
    lateral_distance_nm = pd.to_numeric(
        labeled_pairs["lateral_distance_nm"], errors="coerce"
    )
    vertical_separation_m = pd.to_numeric(
        labeled_pairs["vertical_separation_m"], errors="coerce"
    )
    vertical_separation_limit_m = vertical_separation_ft * METRES_PER_FOOT
    labelable = lateral_distance_nm.notna() & vertical_separation_m.notna()
    separation_breach = (
        (lateral_distance_nm <= lateral_separation_nm)
        & (vertical_separation_m <= vertical_separation_limit_m)
    )

    collision_risk = pd.Series(pd.NA, index=labeled_pairs.index, dtype="Int64")
    collision_risk.loc[labelable] = separation_breach.loc[labelable].astype(int)
    labeled_pairs["collision_risk"] = collision_risk
    labeled_pairs["lateral_separation_limit_nm"] = lateral_separation_nm
    labeled_pairs["vertical_separation_limit_ft"] = vertical_separation_ft
    return labeled_pairs


# Week 2, step 7: confirm the pair table has the required separation features.
def _validate_pair_columns(pairs: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(pairs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Pair data is missing required columns: {missing}.")


# Week 2, step 7: label a saved pair-feature CSV using separation minima.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to a pair-feature CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the labeled CSV (default: data/processed/).",
    )
    parser.add_argument(
        "--lateral-separation-nm",
        type=float,
        default=DEFAULT_LATERAL_SEPARATION_NM,
        help="Lateral separation limit in nautical miles (default: 5).",
    )
    parser.add_argument(
        "--vertical-separation-ft",
        type=float,
        default=DEFAULT_VERTICAL_SEPARATION_FT,
        help="Vertical separation limit in feet (default: 1000).",
    )
    args = parser.parse_args()

    labeled_pairs = label_collision_risk(
        pd.read_csv(args.input_path),
        lateral_separation_nm=args.lateral_separation_nm,
        vertical_separation_ft=args.vertical_separation_ft,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.input_path.stem}_labeled.csv"
    labeled_pairs.to_csv(output_path, index=False)

    print(f"Input pairs     : {len(labeled_pairs)}")
    print(f"Risk labels     : {int(labeled_pairs['collision_risk'].sum())}")
    print(f"Unlabelled pairs: {int(labeled_pairs['collision_risk'].isna().sum())}")
    print(f"Wrote           : {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()