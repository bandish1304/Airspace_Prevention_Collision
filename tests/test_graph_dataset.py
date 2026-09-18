import unittest

import pandas as pd

from src.graph.dataset import GraphSnapshotDataset


class TestGraphSnapshotDataset(unittest.TestCase):
    def test_graph_snapshot_dataset_builds_sequence_by_time(self) -> None:
        states = pd.DataFrame(
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
                    "icao24": "a1",
                    "snapshot_time": 1700000010,
                    "latitude": 33.8117,
                    "longitude": -118.3204,
                    "baro_altitude": 401.0,
                    "velocity": 53.0,
                    "true_track": 123.0,
                    "on_ground": False,
                    "time_position": 1700000010,
                },
                {
                    "icao24": "a3",
                    "snapshot_time": 1700000010,
                    "latitude": 34.5,
                    "longitude": -119.0,
                    "baro_altitude": 3000.0,
                    "velocity": 80.0,
                    "true_track": 200.0,
                    "on_ground": False,
                    "time_position": 1700000010,
                },
            ]
        )

        dataset = GraphSnapshotDataset(states, max_distance_nm=20.0)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["snapshot_time"], 1700000000)
        self.assertIn("graph", dataset[0])
        self.assertEqual(dataset[1]["snapshot_time"], 1700000010)
        self.assertGreaterEqual(dataset[0]["graph"]["num_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
