import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_profile as evaluator
import build_profile as builder


def make_record(prompt_id, workload_id, expert_counts):
    counts = builder.Counter({
        (0, expert): count
        for expert, count in expert_counts.items()
    })
    weights = builder.defaultdict(float)

    for key, count in counts.items():
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
        "counts": counts,
        "weight_sums": weights,
        "activation_count": sum(counts.values()),
    }


class TestRankingMetrics(unittest.TestCase):
    def test_average_ranks_handles_ties(self):
        self.assertEqual(
            evaluator.average_ranks([1.0, 1.0, 3.0]),
            [1.5, 1.5, 3.0],
        )

    def test_spearman_identical_rankings(self):
        self.assertAlmostEqual(
            evaluator.spearman_correlation(
                [0.9, 0.5, 0.1],
                [0.9, 0.5, 0.1],
            ),
            1.0,
        )

    def test_spearman_reverse_rankings(self):
        self.assertAlmostEqual(
            evaluator.spearman_correlation(
                [0.1, 0.5, 0.9],
                [0.9, 0.5, 0.1],
            ),
            -1.0,
        )

    def test_spearman_constant_scores_returns_zero(self):
        self.assertEqual(
            evaluator.spearman_correlation(
                [1.0, 1.0, 1.0],
                [0.9, 0.5, 0.1],
            ),
            0.0,
        )

    def test_perfect_top_k_ranking(self):
        result = evaluator.ranking_metrics(
            [0.9, 0.1, 0.0],
            [0.9, 0.1, 0.0],
            k=1,
            relevant_k=1,
        )
        self.assertEqual(result["precision_at_k"], 1.0)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertAlmostEqual(result["ndcg_at_k"], 1.0)
        self.assertAlmostEqual(result["spearman"], 1.0)

    def test_wrong_top_k_ranking(self):
        result = evaluator.ranking_metrics(
            [0.0, 0.1, 0.9],
            [0.9, 0.1, 0.0],
            k=1,
            relevant_k=1,
        )
        self.assertEqual(result["precision_at_k"], 0.0)
        self.assertEqual(result["recall_at_k"], 0.0)
        self.assertEqual(result["ndcg_at_k"], 0.0)
        self.assertAlmostEqual(result["spearman"], -1.0)

    def test_predicted_ties_use_expert_id_as_tiebreak(self):
        result = evaluator.ranking_metrics(
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            k=1,
            relevant_k=1,
        )
        self.assertEqual(result["hits"], 0)

    def test_tie_aware_metrics_average_boundary_ties(self):
        result = evaluator.ranking_metrics(
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            k=1,
            relevant_k=1,
        )
        self.assertAlmostEqual(
            result["tie_aware_precision_at_k"], 0.5
        )
        self.assertAlmostEqual(
            result["tie_aware_recall_at_k"], 0.5
        )
        self.assertAlmostEqual(
            result["tie_aware_ndcg_at_k"], 0.5
        )

    def test_tie_aware_metrics_match_deterministic_without_ties(self):
        result = evaluator.ranking_metrics(
            [0.9, 0.6, 0.2, 0.1],
            [0.8, 0.7, 0.3, 0.1],
            k=2,
            relevant_k=2,
        )
        self.assertAlmostEqual(
            result["tie_aware_precision_at_k"],
            result["precision_at_k"],
        )
        self.assertAlmostEqual(
            result["tie_aware_recall_at_k"],
            result["recall_at_k"],
        )
        self.assertAlmostEqual(
            result["tie_aware_ndcg_at_k"],
            result["ndcg_at_k"],
        )

    def test_expected_top_k_probabilities_reject_invalid_limit(self):
        with self.assertRaises(ValueError):
            evaluator._expected_top_k_probabilities(
                [0.9, 0.1], 0
            )


class TestEvaluationFold(unittest.TestCase):
    def test_fold_holds_out_entire_session_and_records_support(self):
        sessions = []

        for session_number, workload_id in enumerate(
            ("session-a", "session-b", "session-c")
        ):
            records = []

            for prompt_id in range(1, 5):
                if session_number == 2:
                    counts = {2: 10, 1: 2}
                elif session_number == 1:
                    counts = {1: 10, 2: 2}
                else:
                    counts = {0: 10, 1: 2}

                records.append(
                    make_record(prompt_id, workload_id, counts)
                )

            sessions.append({
                "path": Path(f"{workload_id}.jsonl"),
                "records": records,
                "workload_id": workload_id,
                "activation_count": sum(
                    record["activation_count"] for record in records
                ),
            })

        fold = evaluator.evaluate_fold(
            heldout_index=2,
            paths=[session["path"] for session in sessions],
            loaded_sessions=sessions,
            n_layer=1,
            n_expert=3,
            k=1,
            relevant_k=1,
        )

        self.assertEqual(fold["heldout_session"], "session-c")
        self.assertEqual(
            fold["training_sessions"],
            ["session-a", "session-b"],
        )
        self.assertEqual(
            fold["training_support"]["prompt_count"],
            8,
        )
        self.assertEqual(
            fold["training_support"]["source_workload_count"],
            2,
        )
        self.assertFalse(
            fold["training_support"]["eligible_for_prediction"]
        )
        self.assertEqual(fold["heldout_support"]["prompt_count"], 4)
        self.assertEqual(fold["layer_count"], 1)
        self.assertEqual(len(fold["pwea_layer_metrics"]), 1)
        self.assertEqual(len(fold["baseline_layer_metrics"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
