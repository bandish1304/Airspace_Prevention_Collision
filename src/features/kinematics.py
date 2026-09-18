"""Kinematic feature calculations for pairs of aircraft."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EARTH_RADIUS_METRES = 6_371_008.8


# Week 2, step 1: calculate lateral/horizontal separation between two aircraft.
"""This is a vectorized Python function that uses the Haversine formula 
to calculate the great-circle distance in meters between two geographical points using WGS-84 
coordinates. It converts flexible input types into NumPy arrays, validates them, handles 
the trigonometric math in radians, and dynamically returns either a single float 
or an array depending on the input size"""
def haversine_distance(
    latitude_a: ArrayLike,
    longitude_a: ArrayLike,
    latitude_b: ArrayLike,
    longitude_b: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return the great-circle distance between positions in metres.

    Latitudes and longitudes are decimal degrees in WGS-84 coordinates, as
    returned by OpenSky. Inputs may be scalars or broadcastable NumPy arrays.
    """
    latitude_a_array = np.asarray(latitude_a, dtype=float)
    longitude_a_array = np.asarray(longitude_a, dtype=float)
    latitude_b_array = np.asarray(latitude_b, dtype=float)
    longitude_b_array = np.asarray(longitude_b, dtype=float)

    _validate_coordinates(latitude_a_array, longitude_a_array)
    _validate_coordinates(latitude_b_array, longitude_b_array)

    latitude_a_radians = np.radians(latitude_a_array)
    latitude_b_radians = np.radians(latitude_b_array)
    latitude_difference = latitude_b_radians - latitude_a_radians
    longitude_difference = np.radians(longitude_b_array - longitude_a_array)

    haversine_term = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude_a_radians)
        * np.cos(latitude_b_radians)
        * np.sin(longitude_difference / 2) ** 2
    )
    central_angle = 2 * np.arctan2(
        np.sqrt(haversine_term), np.sqrt(1 - haversine_term)
    )
    distance = EARTH_RADIUS_METRES * central_angle

    return float(distance) if distance.ndim == 0 else distance


# Week 2, step 2: calculate whether two aircraft are converging or separating.
"""This function calculates the radial closing speed between two aircraft 
by projectively mapping their relative velocity vectors onto their 
shared line-of-sight vector using a dot product."""
def closing_speed(
    latitude_a: ArrayLike,
    longitude_a: ArrayLike,
    velocity_a: ArrayLike,
    true_track_a: ArrayLike,
    latitude_b: ArrayLike,
    longitude_b: ArrayLike,
    velocity_b: ArrayLike,
    true_track_b: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return radial closing speed in metres per second.

    A positive result means the aircraft are converging; a negative result
    means they are separating. Velocities are OpenSky ground speeds in m/s,
    and tracks are degrees clockwise from true north.
    """
    latitude_a_array = np.asarray(latitude_a, dtype=float)
    longitude_a_array = np.asarray(longitude_a, dtype=float)
    velocity_a_array = np.asarray(velocity_a, dtype=float)
    true_track_a_array = np.asarray(true_track_a, dtype=float)
    latitude_b_array = np.asarray(latitude_b, dtype=float)
    longitude_b_array = np.asarray(longitude_b, dtype=float)
    velocity_b_array = np.asarray(velocity_b, dtype=float)
    true_track_b_array = np.asarray(true_track_b, dtype=float)

    _validate_coordinates(latitude_a_array, longitude_a_array)
    _validate_coordinates(latitude_b_array, longitude_b_array)
    _validate_kinematics(velocity_a_array, true_track_a_array)
    _validate_kinematics(velocity_b_array, true_track_b_array)

    bearing_a_to_b = _initial_bearing(
        latitude_a_array, longitude_a_array, latitude_b_array, longitude_b_array
    )
    unit_east = np.sin(bearing_a_to_b)
    unit_north = np.cos(bearing_a_to_b)

    track_a_radians = np.radians(true_track_a_array)
    track_b_radians = np.radians(true_track_b_array)
    relative_east = (
        velocity_a_array * np.sin(track_a_radians)
        - velocity_b_array * np.sin(track_b_radians)
    )
    relative_north = (
        velocity_a_array * np.cos(track_a_radians)
        - velocity_b_array * np.cos(track_b_radians)
    )
    speed = relative_east * unit_east + relative_north * unit_north

    same_position = haversine_distance(
        latitude_a_array, longitude_a_array, latitude_b_array, longitude_b_array
    ) == 0
    speed = np.where(same_position, 0.0, speed)
    return float(speed) if speed.ndim == 0 else speed


# Week 2, step 3: calculate the smallest heading difference between aircraft.
def bearing_difference(
    true_track_a: ArrayLike, true_track_b: ArrayLike
) -> float | NDArray[np.float64]:
    """Return the absolute difference between true tracks in degrees.

    The result is always in the range [0, 180], accounting for the wraparound
    between 0 and 360 degrees.
    """
    true_track_a_array = np.asarray(true_track_a, dtype=float)
    true_track_b_array = np.asarray(true_track_b, dtype=float)
    _validate_true_tracks(true_track_a_array)
    _validate_true_tracks(true_track_b_array)

    difference = np.abs((true_track_a_array - true_track_b_array + 180) % 360 - 180)
    return float(difference) if difference.ndim == 0 else difference


# Week 2, step 4: calculate altitude separation between two aircraft.
def vertical_separation(
    altitude_a: ArrayLike, altitude_b: ArrayLike
) -> float | NDArray[np.float64]:
    """Return the absolute vertical separation in metres.

    OpenSky altitudes are in metres. Missing altitude values remain NaN so the
    later pair-processing pipeline can exclude or handle them deliberately.
    """
    altitude_a_array = np.asarray(altitude_a, dtype=float)
    altitude_b_array = np.asarray(altitude_b, dtype=float)
    separation = np.abs(altitude_a_array - altitude_b_array)
    return float(separation) if separation.ndim == 0 else separation


# Week 2, step 5: calculate time until the horizontal closest point of approach.
def time_to_cpa(
    latitude_a: ArrayLike,
    longitude_a: ArrayLike,
    velocity_a: ArrayLike,
    true_track_a: ArrayLike,
    latitude_b: ArrayLike,
    longitude_b: ArrayLike,
    velocity_b: ArrayLike,
    true_track_b: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return time to horizontal CPA in seconds under constant velocity.

    Returns ``numpy.inf`` when the aircraft have no future horizontal CPA,
    including stationary relative motion and pairs that are separating.
    """
    latitude_a_array = np.asarray(latitude_a, dtype=float)
    longitude_a_array = np.asarray(longitude_a, dtype=float)
    velocity_a_array = np.asarray(velocity_a, dtype=float)
    true_track_a_array = np.asarray(true_track_a, dtype=float)
    latitude_b_array = np.asarray(latitude_b, dtype=float)
    longitude_b_array = np.asarray(longitude_b, dtype=float)
    velocity_b_array = np.asarray(velocity_b, dtype=float)
    true_track_b_array = np.asarray(true_track_b, dtype=float)

    _validate_coordinates(latitude_a_array, longitude_a_array)
    _validate_coordinates(latitude_b_array, longitude_b_array)
    _validate_kinematics(velocity_a_array, true_track_a_array)
    _validate_kinematics(velocity_b_array, true_track_b_array)

    position_east, position_north = _relative_position(
        latitude_a_array, longitude_a_array, latitude_b_array, longitude_b_array
    )
    velocity_a_east, velocity_a_north = _velocity_components(
        velocity_a_array, true_track_a_array
    )
    velocity_b_east, velocity_b_north = _velocity_components(
        velocity_b_array, true_track_b_array
    )
    relative_velocity_east = velocity_b_east - velocity_a_east
    relative_velocity_north = velocity_b_north - velocity_a_north
    relative_speed_squared = (
        relative_velocity_east**2 + relative_velocity_north**2
    )
    approach_rate = (
        position_east * relative_velocity_east
        + position_north * relative_velocity_north
    )
    time = np.divide(
        -approach_rate,
        relative_speed_squared,
        out=np.full(np.broadcast(position_east, relative_speed_squared).shape, np.inf),
        where=relative_speed_squared > 0,
    )
    time = np.where(time > 0, time, np.inf)
    return float(time) if time.ndim == 0 else time


# Week 2, steps 2 and 5: calculate the ground-velocity components from a track.
def _velocity_components(
    velocity: NDArray[np.float64], true_track: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    track_radians = np.radians(true_track)
    return velocity * np.sin(track_radians), velocity * np.cos(track_radians)


# Week 2, step 5: calculate the local east/north offset from aircraft A to B.
def _relative_position(
    latitude_a: NDArray[np.float64],
    longitude_a: NDArray[np.float64],
    latitude_b: NDArray[np.float64],
    longitude_b: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    distance = haversine_distance(latitude_a, longitude_a, latitude_b, longitude_b)
    bearing = _initial_bearing(latitude_a, longitude_a, latitude_b, longitude_b)
    return distance * np.sin(bearing), distance * np.cos(bearing)


# Week 2, steps 2 and 5: calculate the direction from the first aircraft to the second.
def _initial_bearing(
    latitude_a: NDArray[np.float64],
    longitude_a: NDArray[np.float64],
    latitude_b: NDArray[np.float64],
    longitude_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    latitude_a_radians = np.radians(latitude_a)
    latitude_b_radians = np.radians(latitude_b)
    longitude_difference = np.radians(longitude_b - longitude_a)
    east = np.sin(longitude_difference) * np.cos(latitude_b_radians)
    north = np.cos(latitude_a_radians) * np.sin(latitude_b_radians) - (
        np.sin(latitude_a_radians)
        * np.cos(latitude_b_radians)
        * np.cos(longitude_difference)
    )
    return np.arctan2(east, north)


# Week 2, steps 1-2: reject coordinates outside valid geographic bounds.
def _validate_coordinates(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> None:
    if np.any((latitudes < -90) | (latitudes > 90)):
        raise ValueError("Latitude must be within [-90, 90] degrees.")
    if np.any((longitudes < -180) | (longitudes > 180)):
        raise ValueError("Longitude must be within [-180, 180] degrees.")


# Week 2, step 2: reject invalid speed and true-track values.
def _validate_kinematics(
    velocities: NDArray[np.float64], tracks: NDArray[np.float64]
) -> None:
    if np.any(velocities < 0):
        raise ValueError("Velocity must be greater than or equal to zero.")
    _validate_true_tracks(tracks)


# Week 2, steps 2-3: reject headings outside OpenSky's true-track range.
def _validate_true_tracks(tracks: NDArray[np.float64]) -> None:
    if np.any((tracks < 0) | (tracks >= 360)):
        raise ValueError("True track must be within [0, 360) degrees.")