import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "build_profile.py"
SPEC = importlib.util.spec_from_file_location("pwea_build_profile", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_record(prompt_id, workload_id, counts):
    counter = MODULE.Counter(counts)
    weights = MODULE.defaultdict(float)

    for key, count in counter.items():
        weights[key] = float(count) * 0.5

    return {
        "prompt_id": prompt_id,
        "workload_id": workload_id,
        "labels": {
            "language": "Python",
            "framework": "none",
            "task_category": "code-generation",
            "workload_phase": "inference",
        },
        "counts": counter,
        "weight_sums": weights,
        "activation_count": sum(counter.values()),
    }


def json_record(prompt_id, dropped=0, layer=0, expert=0):
    return {
        "record_type": "etf_prompt",
        "schema_version": 1,
        "prompt_id": prompt_id,
        "workload": {
            "workload_id": "run-a",
            "language": "Python",
            "framework": "none",
            "task_category": "code-generation",
            "workload_phase": "inference",
        },
        "window": {
            "first_prompt_id": prompt_id,
            "last_prompt_id": prompt_id,
            "completed_prompts": 1,
            "dropped_events": dropped,
        },
        "activations": [{
            "layer": layer,
            "token_position": 0,
            "expert_id": expert,
            "routing_weight": 0.5,
        }],
    }


class TestPWEAProfile(unittest.TestCase):
    def test_per_prompt_normalization_differs_from_aggregate(self):
        records = [
            make_record(1, "run-a", {(0, 0): 9, (0, 1): 1}),
            make_record(2, "run-b", {(0, 1): 1}),
        ]

        profile = MODULE.build_profile(
            records, n_layer=1, n_expert=2,
            min_prompts=2, min_workloads=2,
        )[0]

        experts = {
            item["expert_id"]: item
            for item in profile["layers"][0]["experts"]
        }

        self.assertAlmostEqual(
            experts[0]["mean_prompt_activation_share"], 0.45
        )
        self.assertAlmostEqual(
            experts[1]["mean_prompt_activation_share"], 0.55
        )
        self.assertAlmostEqual(
            experts[0]["aggregate_activation_share"], 9 / 11
        )
        self.assertAlmostEqual(
            experts[1]["aggregate_activation_share"], 2 / 11
        )

    def test_prediction_requires_prompt_and_run_support(self):
        records = [
            make_record(index, "same-run", {(0, 0): 1})
            for index in range(1, 21)
        ]

        profile = MODULE.build_profile(
            records, 1, 1, min_prompts=20, min_workloads=3
        )[0]

        self.assertEqual(profile["support"]["prompt_count"], 20)
        self.assertEqual(profile["support"]["source_workload_count"], 1)
        self.assertFalse(profile["support"]["eligible_for_prediction"])

    def test_prediction_eligible_when_both_thresholds_met(self):
        records = [
            make_record(1, "run-a", {(0, 0): 1}),
            make_record(2, "run-b", {(0, 0): 1}),
            make_record(3, "run-c", {(0, 0): 1}),
        ]

        profile = MODULE.build_profile(
            records, 1, 1, min_prompts=3, min_workloads=3
        )[0]

        self.assertTrue(profile["support"]["eligible_for_prediction"])

    def test_empty_layer_has_zero_shares(self):
        records = [make_record(1, "run-a", {})]

        profile = MODULE.build_profile(
            records, 2, 2, min_prompts=1, min_workloads=1
        )[0]

        for layer in profile["layers"]:
            self.assertEqual(layer["activation_count"], 0)
            self.assertTrue(all(
                expert["mean_prompt_activation_share"] == 0.0
                for expert in layer["experts"]
            ))

    def test_deterministic_expert_tie_break(self):
        records = [
            make_record(1, "run-a", {(0, 0): 1, (0, 1): 1}),
        ]

        profile = MODULE.build_profile(
            records, 1, 2, min_prompts=1, min_workloads=1
        )[0]

        ids = [
            item["expert_id"]
            for item in profile["layers"][0]["experts"]
        ]
        self.assertEqual(ids, [0, 1])

    def test_duplicate_prompt_id_rejected(self):
        rows = [json_record(1), json_record(1)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            with self.assertRaisesRegex(
                MODULE.ValidationError, "duplicate prompt_id"
            ):
                MODULE.load_records(path, 1, 1)

    def test_multiple_files_allow_prompt_ids_to_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "run-a.jsonl"
            second = Path(directory) / "run-b.jsonl"
            first.write_text(json.dumps(json_record(1)) + chr(10))

            row = json_record(1)
            row["workload"]["workload_id"] = "run-b"
            second.write_text(json.dumps(row) + chr(10))

            records, workload_ids, total, sources = (
                MODULE.load_multiple_records([first, second], 1, 1)
            )

            self.assertEqual([r["prompt_id"] for r in records], [1, 1])
            self.assertEqual(workload_ids, {"run-a", "run-b"})
            self.assertEqual(total, 2)
            self.assertEqual(len(sources), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in sources))

    def test_multiple_files_reject_reused_workload_id(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first.write_text(json.dumps(json_record(1)) + chr(10))
            second.write_text(json.dumps(json_record(1)) + chr(10))

            with self.assertRaisesRegex(
                MODULE.ValidationError, "workload_id reused"
            ):
                MODULE.load_multiple_records([first, second], 1, 1)

    def test_multiple_files_reject_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(json.dumps(json_record(1)) + chr(10))

            with self.assertRaisesRegex(
                MODULE.ValidationError, "supplied more than once"
            ):
                MODULE.load_multiple_records([path, path], 1, 1)

    def test_dropped_events_rejected(self):
        row = json_record(1, dropped=1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dropped.jsonl"
            path.write_text(json.dumps(row) + "\n")

            with self.assertRaisesRegex(
                MODULE.ValidationError, "dropped_events"
            ):
                MODULE.load_records(path, 1, 1)

    def test_out_of_range_layer_rejected(self):
        row = json_record(1, layer=1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dimension.jsonl"
            path.write_text(json.dumps(row) + "\n")

            with self.assertRaisesRegex(
                MODULE.ValidationError, "outside model dimensions"
            ):
                MODULE.load_records(path, 1, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
