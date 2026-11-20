import sys
import unittest
from unittest import mock
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_prompt_transfer as evaluator


CATEGORIES = (
    "code-generation",
    "debugging",
    "code-explanation",
    "code-transformation",
    "algorithmic-problem-solving",
)


def make_record(prompt_id, workload_id, category, counts):
    expert_counts = Counter({
        (0, expert): count for expert, count in counts.items()
    })
    weights = defaultdict(float)
    for key, count in expert_counts.items():
        weights[key] = count * 0.5

    return {
        "prompt_id": prompt_id,
        "workload_id": workload_id,
        "labels": {
            "language": "Python",
            "framework": "none",
            "task_category": category,
            "workload_phase": "inference",
        },
        "counts": expert_counts,
        "weight_sums": weights,
        "activation_count": sum(expert_counts.values()),
    }


def make_sessions():
    sessions = []
    for category_index, category in enumerate(CATEGORIES):
        for session_index in range(3):
            workload_id = f"{category}-session-{session_index + 1}"
            records = []
            for prompt_id in range(1, 8):
                preferred = (category_index + session_index + prompt_id) % 4
                secondary = (preferred + 1) % 4
                records.append(make_record(
                    prompt_id, workload_id, category,
                    {preferred: 9, secondary: 1},
                ))
            sessions.append({
                "path": Path(f"{workload_id}.jsonl"),
                "records": records,
                "workload_id": workload_id,
                "task_category": category,
                "activation_count": sum(
                    record["activation_count"] for record in records
                ),
            })
    return sessions


class TestPromptTransfer(unittest.TestCase):
    def setUp(self):
        self.sessions = make_sessions()

    def test_fold_tests_each_prompt_individually(self):
        fold = evaluator.evaluate_category_fold(
            "debugging", self.sessions, CATEGORIES, 1, 4, 1, 1
        )
        self.assertEqual(fold["heldout_prompt_count"], 21)
        self.assertEqual(len(fold["prompt_results"]), 21)
        self.assertTrue(all(
            row["prompt_id"] in range(1, 8)
            for row in fold["prompt_results"]
        ))

    def test_training_excludes_heldout_category(self):
        fold = evaluator.evaluate_category_fold(
            "debugging", self.sessions, CATEGORIES, 1, 4, 1, 1
        )
        self.assertEqual(
            set(fold["training_categories"]),
            set(CATEGORIES) - {"debugging"},
        )
        self.assertEqual(len(fold["training_sessions"]), 12)
        self.assertEqual(len(fold["heldout_sessions"]), 3)

    def test_no_heldout_records_are_mutated(self):
        before = [
            dict(record["labels"])
            for session in self.sessions
            for record in session["records"]
        ]
        evaluator.evaluate_category_fold(
            "debugging", self.sessions, CATEGORIES, 1, 4, 1, 1
        )
        after = [
            record["labels"]
            for session in self.sessions
            for record in session["records"]
        ]
        self.assertEqual(after, before)

    def test_all_five_folds_cover_105_prompts(self):
        result = evaluator.evaluate_prompt_transfer(
            self.sessions, CATEGORIES, 1, 4, 1, 1
        )
        self.assertEqual(result["fold_count"], 5)
        self.assertEqual(result["total_test_prompts"], 105)
        self.assertEqual(
            [fold["heldout_prompt_count"] for fold in result["folds"]],
            [21] * 5,
        )

    def test_each_prompt_result_contains_both_methods(self):
        fold = evaluator.evaluate_category_fold(
            "debugging", self.sessions, CATEGORIES, 1, 4, 1, 1
        )
        for row in fold["prompt_results"]:
            self.assertIn("pwea", row)
            self.assertIn("frequency_baseline", row)
            self.assertEqual(len(row["pwea_layer_metrics"]), 1)
            self.assertEqual(len(row["baseline_layer_metrics"]), 1)

    def test_profile_pair_aligns_scores_by_expert_id(self):
        train_profile = {
            "layers": [{
                "experts": [
                    {"expert_id": 0, "mean_prompt_activation_share": 0.9,
                     "aggregate_activation_share": 0.8},
                    {"expert_id": 1, "mean_prompt_activation_share": 0.05,
                     "aggregate_activation_share": 0.1},
                    {"expert_id": 2, "mean_prompt_activation_share": 0.05,
                     "aggregate_activation_share": 0.1},
                ]
            }]
        }

        # Test profile is sorted by activation, not by expert ID.
        test_profile = {
            "layers": [{
                "experts": [
                    {"expert_id": 1, "mean_prompt_activation_share": 0.8,
                     "aggregate_activation_share": 0.8},
                    {"expert_id": 0, "mean_prompt_activation_share": 0.1,
                     "aggregate_activation_share": 0.1},
                    {"expert_id": 2, "mean_prompt_activation_share": 0.1,
                     "aggregate_activation_share": 0.1},
                ]
            }]
        }

        with mock.patch.object(
            evaluator.builder, "build_profile", return_value=[test_profile]
        ):
            result = evaluator._profile_pair_metrics(
                train_profile,
                {"activation_count": 10},
                n_layer=1,
                k=1,
                relevant_k=1,
            )

        # Training predicts expert 0, while the held-out prompt activates
        # expert 1 most. Correct ID alignment must therefore score zero.
        self.assertEqual(result["pwea"]["precision_at_k"], 0.0)
        self.assertEqual(result["pwea"]["recall_at_k"], 0.0)
        self.assertAlmostEqual(result["pwea"]["ndcg_at_k"], 0.125)
        self.assertEqual(
            result["frequency_baseline"]["precision_at_k"], 0.0
        )

    def test_rejects_session_with_wrong_prompt_count(self):
        self.sessions[0]["records"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 7 prompts"):
            evaluator.evaluate_category_fold(
                "debugging", self.sessions, CATEGORIES, 1, 4, 1, 1
            )

    def test_rejects_duplicate_prompt_ids_within_session(self):
        heldout = next(
            session for session in self.sessions
            if session["task_category"] == "debugging"
        )
        heldout["records"][1]["prompt_id"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate prompt_id"):
            evaluator.evaluate_category_fold(
                "debugging", self.sessions, CATEGORIES, 1, 4, 1, 1
            )

    def test_rejects_unknown_category(self):
        with self.assertRaises(ValueError):
            evaluator.evaluate_category_fold(
                "unknown", self.sessions, CATEGORIES, 1, 4, 1, 1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
