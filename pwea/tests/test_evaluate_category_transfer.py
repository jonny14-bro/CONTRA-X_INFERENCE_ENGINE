import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_category_transfer as evaluator


CATEGORIES = (
    "code-generation",
    "debugging",
    "code-explanation",
    "code-transformation",
    "algorithmic-problem-solving",
)


def make_record(prompt_id, workload_id, category, expert_counts):
    from collections import Counter, defaultdict

    counts = Counter({
        (0, expert): count
        for expert, count in expert_counts.items()
    })
    weights = defaultdict(float)

    for key, count in counts.items():
        weights[key] = float(count) * 0.5

    return {
        "prompt_id": prompt_id,
        "workload_id": workload_id,
        "labels": {
            "language": "Python",
            "framework": "none",
            "task_category": category,
            "workload_phase": "inference",
        },
        "counts": counts,
        "weight_sums": weights,
        "activation_count": sum(counts.values()),
    }


def make_sessions():
    sessions = []

    for category_index, category in enumerate(CATEGORIES):
        for session_index in range(3):
            workload_id = f"{category}-session-{session_index + 1}"
            records = []

            for prompt_id in range(1, 8):
                # Different categories have deliberately different
                # expert distributions to make category leakage visible.
                preferred = (category_index + session_index) % 4
                secondary = (preferred + 1) % 4

                records.append(
                    make_record(
                        prompt_id,
                        workload_id,
                        category,
                        {preferred: 9, secondary: 1},
                    )
                )

            sessions.append({
                "path": Path(f"{workload_id}.jsonl"),
                "records": records,
                "workload_id": workload_id,
                "activation_count": sum(
                    record["activation_count"] for record in records
                ),
                "task_category": category,
            })

    return sessions


class TestCategoryTransferExpertAlignment(unittest.TestCase):
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

        # Deliberately sorted differently from the training profile.
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

        result = evaluator._evaluate_profile_pair(
            train_profile,
            test_profile,
            n_layer=1,
            k=1,
            relevant_k=1,
        )

        self.assertEqual(result["pwea"]["precision_at_k"], 0.0)
        self.assertEqual(result["pwea"]["recall_at_k"], 0.0)
        self.assertAlmostEqual(result["pwea"]["ndcg_at_k"], 0.125)
        self.assertEqual(
            result["frequency_baseline"]["precision_at_k"], 0.0
        )

    def test_duplicate_expert_ids_are_rejected(self):
        experts = [
            {"expert_id": 0, "score": 0.8},
            {"expert_id": 0, "score": 0.2},
        ]

        with self.assertRaisesRegex(ValueError, "expert IDs"):
            evaluator._scores_by_expert_id(experts, "score")


class TestCategoryTransferFold(unittest.TestCase):
    def setUp(self):
        self.sessions = make_sessions()

    def test_excludes_every_heldout_category_session(self):
        fold = evaluator.evaluate_category_fold(
            heldout_category="debugging",
            loaded_sessions=self.sessions,
            n_layer=1,
            n_expert=4,
            k=1,
            relevant_k=1,
        )

        self.assertEqual(
            fold["heldout_category"],
            "debugging",
        )
        self.assertEqual(
            len(fold["heldout_sessions"]),
            3,
        )
        self.assertEqual(
            len(fold["training_sessions"]),
            12,
        )
        training_ids = set(fold["training_sessions"])
        debugging_ids = {
            session["workload_id"]
            for session in self.sessions
            if session["task_category"] == "debugging"
        }
        self.assertTrue(training_ids.isdisjoint(debugging_ids))

    def test_training_pool_contains_only_other_four_categories(self):
        fold = evaluator.evaluate_category_fold(
            "debugging", self.sessions, 1, 4, 1, 1
        )

        self.assertEqual(
            set(fold["training_categories"]),
            set(CATEGORIES) - {"debugging"},
        )

    def test_training_and_heldout_support_are_recorded(self):
        fold = evaluator.evaluate_category_fold(
            "debugging", self.sessions, 1, 4, 1, 1
        )

        self.assertEqual(fold["training_support"]["prompt_count"], 84)
        self.assertEqual(
            fold["training_support"]["source_workload_count"], 12
        )
        self.assertTrue(
            fold["training_support"]["eligible_for_prediction"]
        )

        self.assertEqual(fold["heldout_support"]["prompt_count"], 21)
        self.assertEqual(
            fold["heldout_support"]["source_workload_count"], 3
        )
        self.assertTrue(
            fold["heldout_support"]["eligible_for_prediction"]
        )

    def test_original_records_are_not_mutated(self):
        before = [
            dict(record["labels"])
            for session in self.sessions
            for record in session["records"]
        ]

        evaluator.evaluate_category_fold(
            "debugging", self.sessions, 1, 4, 1, 1
        )

        after = [
            record["labels"]
            for session in self.sessions
            for record in session["records"]
        ]

        self.assertEqual(after, before)

    def test_every_category_can_be_held_out(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                fold = evaluator.evaluate_category_fold(
                    category, self.sessions, 1, 4, 1, 1
                )
                self.assertEqual(fold["heldout_category"], category)
                self.assertEqual(len(fold["heldout_sessions"]), 3)
                self.assertEqual(len(fold["training_sessions"]), 12)

    def test_unknown_heldout_category_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluator.evaluate_category_fold(
                "unknown-category", self.sessions, 1, 4, 1, 1
            )

    def test_inconsistent_non_category_labels_are_rejected(self):
        self.sessions[0]["records"][0]["labels"]["language"] = "Java"

        with self.assertRaises(ValueError):
            evaluator.evaluate_category_fold(
                "debugging", self.sessions, 1, 4, 1, 1
            )

    def test_duplicate_workload_ids_are_rejected(self):
        self.sessions[1]["workload_id"] = self.sessions[0]["workload_id"]

        with self.assertRaises(ValueError):
            evaluator.evaluate_category_fold(
                "debugging", self.sessions, 1, 4, 1, 1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
