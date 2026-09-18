"""Build pair-level kinematic features from an OpenSky snapshot."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .kinematics import (
    bearing_difference,
    closing_speed,
    haversine_distance,
    time_to_cpa,
    vertical_separation,
)

METRES_PER_NAUTICAL_MILE = 1_852.0
DEFAULT_MAX_DISTANCE_NM = 50.0
DEFAULT_MAX_POSITION_AGE_SECONDS = 15.0
REQUIRED_COLUMNS = {
    "icao24",
    "latitude",
    "longitude",
    "on_ground",
    "snapshot_time",
    "time_position",
    "velocity",
    "true_track",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# Week 2, step 6: build features for all unique aircraft pairs within range.
def build_pair_features(
    states: pd.DataFrame,
    max_distance_nm: float = DEFAULT_MAX_DISTANCE_NM,
    max_position_age_seconds: float = DEFAULT_MAX_POSITION_AGE_SECONDS,
) -> pd.DataFrame:
    """Return one feature row per usable aircraft pair within the distance limit."""
    if max_distance_nm <= 0:
        raise ValueError("max_distance_nm must be greater than zero.")
    if max_position_age_seconds < 0:
        raise ValueError("max_position_age_seconds must be greater than or equal to zero.")

    usable_states = _filter_usable_states(states, max_position_age_seconds)
    maximum_distance_metres = max_distance_nm * METRES_PER_NAUTICAL_MILE
    pairs: list[dict[str, object]] = []

    for (_, aircraft_a), (_, aircraft_b) in combinations(usable_states.iterrows(), 2):
        lateral_distance_metres = haversine_distance(
            aircraft_a.latitude,
            aircraft_a.longitude,
            aircraft_b.latitude,
            aircraft_b.longitude,
        )
        if lateral_distance_metres > maximum_distance_metres:
            continue

        altitude_a = _select_altitude(aircraft_a)
        altitude_b = _select_altitude(aircraft_b)
        pairs.append(
            {
                "snapshot_time": aircraft_a.snapshot_time,
                "icao24_a": aircraft_a.icao24,
                "icao24_b": aircraft_b.icao24,
                "position_age_a_s": aircraft_a.position_age_s,
                "position_age_b_s": aircraft_b.position_age_s,
                "latitude_a": aircraft_a.latitude,
                "longitude_a": aircraft_a.longitude,
                "altitude_a_m": altitude_a,
                "velocity_a_mps": aircraft_a.velocity,
                "true_track_a_deg": aircraft_a.true_track,
                "latitude_b": aircraft_b.latitude,
                "longitude_b": aircraft_b.longitude,
                "altitude_b_m": altitude_b,
                "velocity_b_mps": aircraft_b.velocity,
                "true_track_b_deg": aircraft_b.true_track,
                "lateral_distance_m": lateral_distance_metres,
                "lateral_distance_nm": lateral_distance_metres / METRES_PER_NAUTICAL_MILE,
                "closing_speed_mps": closing_speed(
                    aircraft_a.latitude,
                    aircraft_a.longitude,
                    aircraft_a.velocity,
                    aircraft_a.true_track,
                    aircraft_b.latitude,
                    aircraft_b.longitude,
                    aircraft_b.velocity,
                    aircraft_b.true_track,
                ),
                "bearing_difference_deg": bearing_difference(
                    aircraft_a.true_track, aircraft_b.true_track
                ),
                "vertical_separation_m": vertical_separation(altitude_a, altitude_b),
                "time_to_cpa_s": time_to_cpa(
                    aircraft_a.latitude,
                    aircraft_a.longitude,
                    aircraft_a.velocity,
                    aircraft_a.true_track,
                    aircraft_b.latitude,
                    aircraft_b.longitude,
                    aircraft_b.velocity,
                    aircraft_b.true_track,
                ),
            }
        )

    return pd.DataFrame(pairs, columns=_pair_feature_columns())


# Week 2, step 6: remove aircraft without fresh, airborne kinematic data.
def _filter_usable_states(
    states: pd.DataFrame, max_position_age_seconds: float
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(states.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"State data is missing required columns: {missing}.")

    prepared_states = states.copy()
    numeric_columns = [
        "latitude",
        "longitude",
        "snapshot_time",
        "time_position",
        "velocity",
        "true_track",
    ]
    for column in numeric_columns:
        prepared_states[column] = pd.to_numeric(prepared_states[column], errors="coerce")

    prepared_states["position_age_s"] = (
        prepared_states["snapshot_time"] - prepared_states["time_position"]
    )
    required_values = ["icao24", *numeric_columns]
    usable_states = prepared_states.dropna(subset=required_values)
    usable_states = usable_states.loc[
        (~usable_states["on_ground"])
        & (usable_states["position_age_s"] >= 0)
        & (usable_states["position_age_s"] <= max_position_age_seconds)
    ]
    return usable_states.drop_duplicates(subset="icao24", keep="last").reset_index(drop=True)


# Week 2, steps 4 and 6: prefer barometric altitude and fall back to geo altitude.
def _select_altitude(aircraft: pd.Series) -> float:
    baro_altitude = pd.to_numeric(aircraft.get("baro_altitude"), errors="coerce")
    if pd.notna(baro_altitude):
        return float(baro_altitude)
    return float(pd.to_numeric(aircraft.get("geo_altitude"), errors="coerce"))


# Week 2, step 6: keep an explicit, stable schema for empty and populated outputs.
def _pair_feature_columns() -> list[str]:
    return [
        "snapshot_time",
        "icao24_a",
        "icao24_b",
        "position_age_a_s",
        "position_age_b_s",
        "latitude_a",
        "longitude_a",
        "altitude_a_m",
        "velocity_a_mps",
        "true_track_a_deg",
        "latitude_b",
        "longitude_b",
        "altitude_b_m",
        "velocity_b_mps",
        "true_track_b_deg",
        "lateral_distance_m",
        "lateral_distance_nm",
        "closing_speed_mps",
        "bearing_difference_deg",
        "vertical_separation_m",
        "time_to_cpa_s",
    ]


# Week 2, step 6: run the pair-feature pipeline on a saved OpenSky CSV.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to an OpenSky snapshot CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the pair-level CSV (default: data/processed/).",
    )
    parser.add_argument(
        "--max-distance-nm",
        type=float,
        default=DEFAULT_MAX_DISTANCE_NM,
        help="Maximum lateral separation to retain (default: 50).",
    )
    parser.add_argument(
        "--max-position-age-seconds",
        type=float,
        default=DEFAULT_MAX_POSITION_AGE_SECONDS,
        help="Maximum allowed position age (default: 15).",
    )
    args = parser.parse_args()

    pairs = build_pair_features(
        pd.read_csv(args.input_path),
        max_distance_nm=args.max_distance_nm,
        max_position_age_seconds=args.max_position_age_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.input_path.stem}_pairs.csv"
    pairs.to_csv(output_path, index=False)

    print(f"Input aircraft  : {len(pd.read_csv(args.input_path))}")
    print(f"Nearby pairs    : {len(pairs)}")
    print(f"Wrote           : {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()