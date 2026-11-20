import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

MODULE_PATH = PROJECT_DIR / "run_category_transfer.py"
SPEC = importlib.util.spec_from_file_location("pwea_category_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_records(count, workload_id, category):
    records = []
    for prompt_id in range(1, count + 1):
        records.append({
            "prompt_id": prompt_id,
            "workload_id": workload_id,
            "labels": {
                "language": "Python",
                "framework": "none",
                "task_category": category,
                "workload_phase": "inference",
            },
            "counts": {},
            "weight_sums": {},
            "activation_count": 0,
        })
    return records


class TestCategoryRunnerSessionValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.input_path = self.base / "session.jsonl"
        self.input_path.write_text("synthetic\n", encoding="utf-8")

        self.settings = {
            "sessions": [{
                "path": self.input_path,
                "session_id": "test-session-1",
                "task_category": "debugging",
            }],
            "n_layer": 48,
            "n_expert": 128,
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_load_sessions(self, records):
        workload_ids = {record["workload_id"] for record in records}
        activation_count = sum(
            record["activation_count"] for record in records
        )
        with patch.object(
            MODULE.builder,
            "load_records",
            return_value=(records, workload_ids, activation_count),
        ):
            return MODULE.load_sessions(self.settings)

    def test_accepts_exactly_seven_prompt_records(self):
        records = make_records(7, "test-session-1", "debugging")

        sessions, source_files = self.run_load_sessions(records)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["prompt_count"], 7)
        self.assertEqual(source_files[0]["prompt_records"], 7)

    def test_rejects_six_prompt_records(self):
        records = make_records(6, "test-session-1", "debugging")

        with self.assertRaisesRegex(
            MODULE.ConfigError, "expected exactly 7 prompt records, found 6"
        ):
            self.run_load_sessions(records)

    def test_rejects_eight_prompt_records(self):
        records = make_records(8, "test-session-1", "debugging")

        with self.assertRaisesRegex(
            MODULE.ConfigError, "expected exactly 7 prompt records, found 8"
        ):
            self.run_load_sessions(records)

    def test_rejects_category_mismatch(self):
        records = make_records(7, "test-session-1", "code-generation")

        with self.assertRaisesRegex(
            MODULE.ConfigError, "configured category does not match"
        ):
            self.run_load_sessions(records)

    def test_rejects_workload_id_mismatch(self):
        records = make_records(7, "different-session", "debugging")

        with self.assertRaisesRegex(
            MODULE.ConfigError, "configured session_id does not match"
        ):
            self.run_load_sessions(records)


if __name__ == "__main__":
    unittest.main()
