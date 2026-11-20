import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor.inner_validation import build_inner_folds


CATEGORIES = (
    "debugging",
    "code-explanation",
    "code-transformation",
    "algorithmic-problem-solving",
)


def make_samples(categories=CATEGORIES):
    samples = []

    for category in categories:
        for session_number in range(1, 6):
            session_id = f"{category}-s{session_number:02d}"

            for prompt_id in range(1, 8):
                samples.append({
                    "sample_id": f"{session_id}:prompt-{prompt_id:02d}",
                    "session_id": session_id,
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt_text_status": "verified",
                })

    return samples


class TestInnerValidation(unittest.TestCase):

    def setUp(self):
        self.samples = make_samples()
        self.folds = build_inner_folds(self.samples, CATEGORIES)

    def test_creates_five_folds(self):
        self.assertEqual(len(self.folds), 5)

    def test_expected_train_and_validation_sizes(self):
        for fold in self.folds:
            self.assertEqual(len(fold["train_indices"]), 112)
            self.assertEqual(len(fold["validation_indices"]), 28)
            self.assertEqual(len(fold["train_session_ids"]), 16)
            self.assertEqual(len(fold["validation_session_ids"]), 4)

    def test_sessions_never_cross_train_validation_boundary(self):
        for fold in self.folds:
            self.assertTrue(
                set(fold["train_session_ids"]).isdisjoint(
                    fold["validation_session_ids"]
                )
            )

    def test_each_validation_fold_holds_one_session_per_category(self):
        for fold in self.folds:
            heldout = [
                self.samples[index]
                for index in fold["validation_indices"]
            ]

            for category in CATEGORIES:
                category_rows = [
                    row for row in heldout
                    if row["category"] == category
                ]

                self.assertEqual(len(category_rows), 7)
                self.assertEqual(
                    len({row["session_id"] for row in category_rows}),
                    1,
                )

    def test_every_sample_is_validated_exactly_once(self):
        validation_indices = [
            index
            for fold in self.folds
            for index in fold["validation_indices"]
        ]

        self.assertEqual(len(validation_indices), len(self.samples))
        self.assertEqual(
            sorted(validation_indices),
            list(range(len(self.samples))),
        )

    def test_training_and_validation_indices_cover_dataset(self):
        for fold in self.folds:
            combined = (
                fold["train_indices"] + fold["validation_indices"]
            )
            self.assertEqual(sorted(combined), list(range(len(self.samples))))

    def test_rejects_unverified_text(self):
        samples = make_samples()
        samples[0]["prompt_text_status"] = "unresolved"

        with self.assertRaisesRegex(ValueError, "unverified"):
            build_inner_folds(samples, CATEGORIES)

    def test_rejects_missing_prompt(self):
        samples = make_samples()
        samples.pop(0)

        with self.assertRaisesRegex(ValueError, "exactly 7 prompts"):
            build_inner_folds(samples, CATEGORIES)

    def test_rejects_duplicate_prompt_id(self):
        samples = make_samples()
        samples[1]["prompt_id"] = 1

        with self.assertRaisesRegex(ValueError, "prompt IDs"):
            build_inner_folds(samples, CATEGORIES)

    def test_rejects_wrong_session_count(self):
        samples = make_samples()
        samples = [
            row for row in samples
            if row["session_id"] != "debugging-s05"
        ]

        with self.assertRaisesRegex(ValueError, "expected 5 sessions"):
            build_inner_folds(samples, CATEGORIES)

    def test_rejects_unexpected_category(self):
        samples = make_samples()
        samples[0]["category"] = "unknown"

        with self.assertRaisesRegex(ValueError, "unexpected category"):
            build_inner_folds(samples, CATEGORIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
