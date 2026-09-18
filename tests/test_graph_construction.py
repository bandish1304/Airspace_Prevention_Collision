import unittest

import pandas as pd

from src.graph.construction import build_snapshot_graph, validate_snapshot_graph


class TestGraphConstruction(unittest.TestCase):
    def test_build_snapshot_graph_creates_expected_nodes_and_edges(self) -> None:
        snapshot = pd.DataFrame(
            [
                {
                    "icao24": "a1",
                    "snapshot_time": 1700000000,
                    "latitude": 33.8116,
                    "longitude": -118.3203,
                    "baro_altitude": 396.24,
                    "velocity": 52.52,
                    "true_track": 122.6,
                    "on_ground": False,
                    "time_position": 1700000000,
                },
                {
                    "icao24": "a2",
                    "snapshot_time": 1700000000,
                    "latitude": 33.8122,
                    "longitude": -118.3195,
                    "baro_altitude": 400.0,
                    "velocity": 55.0,
                    "true_track": 120.0,
                    "on_ground": False,
                    "time_position": 1700000000,
                },
                {
                    "icao24": "a3",
                    "snapshot_time": 1700000000,
                    "latitude": 34.5,
                    "longitude": -119.0,
                    "baro_altitude": 3000.0,
                    "velocity": 80.0,
                    "true_track": 200.0,
                    "on_ground": False,
                    "time_position": 1700000000,
                },
            ]
        )

        graph = build_snapshot_graph(snapshot, max_distance_nm=20.0)

        self.assertEqual(graph["num_nodes"], 3)
        self.assertEqual(graph["num_edges"], 1)
        self.assertEqual(set(graph["node_index"].keys()), {"a1", "a2", "a3"})
        self.assertIn((0, 1), graph["edge_index"])
        self.assertGreater(graph["edges"][0]["attributes"]["lateral_distance_m"], 0)

    def test_validate_snapshot_graph_reports_valid_structure(self) -> None:
        snapshot = pd.DataFrame(
            [
                {
                    "icao24": "a1",
                    "snapshot_time": 1700000000,
                    "latitude": 33.8116,
                    "longitude": -118.3203,
                    "baro_altitude": 396.24,
                    "velocity": 52.52,
                    "true_track": 122.6,
                    "on_ground": False,
                    "time_position": 1700000000,
                },
                {
                    "icao24": "a2",
                    "snapshot_time": 1700000000,
                    "latitude": 33.8122,
                    "longitude": -118.3195,
                    "baro_altitude": 400.0,
                    "velocity": 55.0,
                    "true_track": 120.0,
                    "on_ground": False,
                    "time_position": 1700000000,
                },
            ]
        )

        validation = validate_snapshot_graph(snapshot, max_distance_nm=20.0)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["num_nodes"], 2)
        self.assertEqual(validation["num_edges"], 1)


if __name__ == "__main__":
    unittest.main()
