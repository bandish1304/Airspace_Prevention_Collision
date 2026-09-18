"""Feature engineering utilities for aircraft-pair risk prediction."""

from .kinematics import (
	bearing_difference,
	closing_speed,
	haversine_distance,
	time_to_cpa,
	vertical_separation,
)
__all__ = [
	"bearing_difference",
	"closing_speed",
	"haversine_distance",
	"time_to_cpa",
	"vertical_separation",
]