"""CPU checks for the bundled S3F inference dependency."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

from common_io import sha256_file, sha256_files


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "model_adapters/s3f_runtime"
spec = importlib.util.spec_from_file_location(
    "bundled_s3f_bridge_test", RUNTIME / "score_s3f_fitness.py"
)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


class S3FRuntimeTests(unittest.TestCase):
    def test_release_source_checksums(self):
        model = json.loads((ROOT / "models/s3f/model.json").read_text())
        self.assertEqual(sha256_file(RUNTIME / "score_s3f_fitness.py"), model["s3f_script_sha256"])
        self.assertEqual(sha256_files(list(RUNTIME.rglob("*.py"))), model["s3f_code_sha256"])
        self.assertEqual(sha256_file(RUNTIME / "config/evaluate/s3f.yaml"), model["s3f_config_sha256"])

    def test_substitution_coordinates(self):
        mutations = bridge.parse_mutations("D3A:A1G", "ACDE")
        self.assertEqual(bridge.format_mutations(mutations), "A1G:D3A")
        self.assertEqual([m.index for m in mutations], [0, 2])
        with self.assertRaises(ValueError):
            bridge.parse_mutations("C1G", "ACDE")

    def test_sequence_windows(self):
        self.assertEqual(bridge._get_optimal_window(0, 100, 1022), (0, 100))
        self.assertEqual(bridge._get_optimal_window(0, 1187, 1022), (0, 1022))
        self.assertEqual(bridge._get_optimal_window(1186, 1187, 1022), (165, 1187))

    def test_entry_points_and_configuration(self):
        self.assertTrue(callable(bridge.DirectedEvolutionFitnessPredictor))
        self.assertTrue(callable(bridge._write_surface_from_pdb))
        self.assertEqual(Path(bridge._default_config_path("s3f")), RUNTIME / "config/evaluate/s3f.yaml")


if __name__ == "__main__":
    unittest.main()
