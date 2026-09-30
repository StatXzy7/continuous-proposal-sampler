import math
import unittest

import numpy as np

from coupled_batch_sampler import AffineMean, sample_coupled_batches, sample_target_baseline

DISTRIBUTION_SAMPLE_SIZE = 10_000
BURN_IN = 45
MEAN_TOLERANCE = 0.05
COVARIANCE_TOLERANCE = 0.12
CDF_TOLERANCE = 0.04
TEST_CDF_PROBABILITIES = (0.10, 0.50, 0.90)


def standard_normal_cdf(value):
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def standard_normal_quantile(probability):
    if probability <= 0 or probability >= 1:
        raise ValueError("quantile probability must be in (0, 1)")
    lower, upper = -10.0, 10.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if standard_normal_cdf(middle) < probability:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)

class CountingMean(AffineMean):
    def __init__(self, mean):
        super().__init__(mean.a, mean.b)
        self.calls = []

    def __call__(self, states):
        states = np.asarray(states, dtype=np.float64)
        self.calls.append(states.shape[0])
        return super().__call__(states)


class SamplerValidationTest(unittest.TestCase):
    def test_rejects_invalid_parameters_before_generation(self):
        target = AffineMean(0.5)
        proposal = AffineMean(0.4)

        cases = [
            dict(initial_states=[[np.nan]], target_mean=target, proposal_mean=proposal,
                 variance=1.0, n_steps=1, lookahead=1),
            dict(initial_states=[[0.0, 0.0]], target_mean=target, proposal_mean=proposal,
                 variance=1.0, n_steps=1, lookahead=1),
            dict(initial_states=[0.0], target_mean=target,
                 proposal_mean=AffineMean(np.eye(2)), variance=1.0, n_steps=1, lookahead=1),
            dict(initial_states=[0.0], target_mean=target, proposal_mean=proposal,
                 variance=0.0, n_steps=1, lookahead=1),
            dict(initial_states=[0.0], target_mean=target, proposal_mean=proposal,
                 variance=-1.0, n_steps=1, lookahead=1),
            dict(initial_states=[0.0], target_mean=target, proposal_mean=proposal,
                 variance=float("nan"), n_steps=1, lookahead=1),
            dict(initial_states=[0.0], target_mean=target, proposal_mean=proposal,
                 variance=1.0, n_steps=0, lookahead=1),
            dict(initial_states=[0.0], target_mean=target, proposal_mean=proposal,
                 variance=1.0, n_steps=1, lookahead=0),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    sample_coupled_batches(**kwargs)

        with self.assertRaises(ValueError):
            AffineMean(np.eye(17))
        with self.assertRaises(ValueError):
            AffineMean(np.eye(2), [0.0, np.inf])
        with self.assertRaises(ValueError):
            sample_target_baseline([0.0], target, -0.1, 1)


class SamplerBehaviorTest(unittest.TestCase):
    def test_seed_reproduces_trajectories_and_accounting(self):
        target = AffineMean(np.full(3, 0.7), [0.1, -0.2, 0.2])
        proposal = AffineMean(np.full(3, 0.7), [0.5, 0.3, -0.4])
        init = np.zeros((6, 3))
        kwargs = dict(variance=0.3, n_steps=17, lookahead=4)

        first = sample_coupled_batches(init, target, proposal, seed=777, **kwargs)
        second = sample_coupled_batches(init, target, proposal, seed=777, **kwargs)
        different = sample_coupled_batches(init, target, proposal, seed=778, **kwargs)

        np.testing.assert_array_equal(first.trajectories, second.trajectories)
        np.testing.assert_array_equal(
            np.concatenate(first.accept_lengths), np.concatenate(second.accept_lengths)
        )
        np.testing.assert_array_equal(
            np.concatenate(first.rejection_positions),
            np.concatenate(second.rejection_positions),
        )
        self.assertFalse(np.array_equal(first.trajectories, different.trajectories))

    def test_identical_mean_accepts_every_draft_unchanged(self):
        target = AffineMean(np.full(3, 0.65), [0.2, -0.3, 0.1])
        init = np.zeros((5, 3))
        _ = sample_target_baseline(init, CountingMean(target), 0.28, 13, seed=42)

        counted_target = CountingMean(target)
        result = sample_coupled_batches(
            init, counted_target, target, 0.28, 13, 5, seed=42
        )
        self.assertEqual(counted_target.calls, [25, 25, 15])
        self.assertEqual(len(counted_target.calls), 3)

        expected = np.empty_like(result.trajectories)
        expected[:, 0, :] = init
        rng = np.random.default_rng(np.random.SeedSequence(42))
        current = init.copy()
        for start, block_length in ((0, 5), (5, 5), (10, 3)):
            drafts = np.empty((5, block_length, 3))
            predecessor = current
            for j in range(block_length):
                drafts[:, j, :] = target(predecessor) + np.sqrt(0.28) * rng.standard_normal((5, 3))
                predecessor = drafts[:, j, :]
            expected[:, start + 1 : start + block_length + 1, :] = drafts
            current = drafts[:, -1, :]
            _ = 1.0 - rng.random((5, block_length))
        np.testing.assert_allclose(result.trajectories, expected, rtol=0.0, atol=1e-14)
        for lengths in result.accept_lengths:
            np.testing.assert_array_equal(lengths, [5, 5, 3])
        for positions in result.rejection_positions:
            np.testing.assert_array_equal(positions, [-1, -1, -1])
        self.assertEqual(result.target_evaluation_count, 5 * 13)
        self.assertEqual(result.target_batch_count, 3)
        np.testing.assert_array_equal(
            result.target_evaluations_per_chain, np.full(5, 13, dtype=np.int64)
        )

    def test_first_rejection_is_reflected_and_later_drafts_are_discarded(self):
        target = AffineMean(np.full(3, 0.7), [0.1, -0.2, 0.2])
        proposal = AffineMean(np.full(3, 0.7), [0.5, 0.3, -0.4])
        init = np.zeros(3)
        result = sample_coupled_batches(init, target, proposal, 0.3, 12, 4, seed=0)

        lengths = result.accept_lengths[0]
        positions = result.rejection_positions[0]
        self.assertEqual(lengths[0], 1)
        self.assertEqual(positions[0], 1)

        rng = np.random.default_rng(np.random.SeedSequence(0))
        first_proposal = np.full(3, proposal.b, dtype=np.float64)
        first_draft = first_proposal + np.sqrt(0.3) * rng.standard_normal(3)
        second_proposal = proposal(first_draft)
        second_target = target(first_draft)
        second_draft = second_proposal + np.sqrt(0.3) * rng.standard_normal(3)
        delta = second_proposal - second_target
        reflected = second_draft - 2.0 * np.dot(
            second_draft - 0.5 * (second_target + second_proposal), delta
        ) / np.dot(delta, delta) * delta

        np.testing.assert_allclose(result.trajectories[0, 1], first_draft)
        np.testing.assert_allclose(result.trajectories[0, 2], reflected)
        self.assertFalse(np.allclose(result.trajectories[0, 2], second_draft))

    def test_first_position_rejection_tail_does_not_over_submit_and_separates_counts(self):
        target = AffineMean(0.0)
        proposal = AffineMean(0.0, 10.0)
        init = np.zeros((3, 1))
        result = sample_coupled_batches(init, target, proposal, 1e-4, 5, 4, seed=3)

        self.assertEqual(result.trajectories.shape, (3, 6, 1))
        for lengths, positions in zip(result.accept_lengths, result.rejection_positions):
            np.testing.assert_array_equal(lengths, [0, 0, 0, 0, 0])
            np.testing.assert_array_equal(positions, [0, 0, 0, 0, 0])
        self.assertEqual(result.target_evaluation_count, 42)
        self.assertEqual(result.target_batch_count, 5)
        np.testing.assert_array_equal(
            result.target_evaluations_per_chain, np.full(3, 14, dtype=np.int64)
        )

    def test_tiny_variance_uses_log_domain_and_remains_finite(self):
        target = AffineMean(0.0)
        proposal = AffineMean(0.0, 1.0)
        result = sample_coupled_batches([[0.0]], target, proposal, 1e-300, 2, 2, seed=1)
        self.assertTrue(np.all(np.isfinite(result.trajectories)))
        np.testing.assert_allclose(result.trajectories[0, 1, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(result.trajectories[0, 2, 0], 0.0, atol=1e-12)


class StationaryDistributionTest(unittest.TestCase):

    def test_dimensions_one_and_sixteen_and_tail_smaller_than_lookahead(self):
        for dim in (1, 16):
            target = AffineMean(0.2 * np.eye(dim), np.linspace(-0.2, 0.2, dim))
            proposal = AffineMean(0.1 * np.eye(dim), np.linspace(0.1, 0.4, dim))
            init = np.zeros((4, dim))
            result = sample_coupled_batches(
                init, target, proposal, 0.2, n_steps=3, lookahead=8, seed=123
            )
            self.assertEqual(result.trajectories.shape, (4, 4, dim))
            for lengths, positions in zip(result.accept_lengths, result.rejection_positions):
                self.assertLessEqual(int(lengths[0]), 3)
                submitted = int(np.sum(lengths + (positions >= 0).astype(np.int64)))
                self.assertEqual(submitted, 3)
            self.assertGreaterEqual(result.target_evaluation_count, 4 * 3)

    def test_analytic_mean_covariance_and_directional_cdfs(self):
        a = 0.72
        variance = 0.25
        target_b = np.array([0.15, -0.20, 0.30])
        proposal_b = np.array([0.70, 0.20, -0.50])
        target = AffineMean(np.full(3, a), target_b)
        proposal = AffineMean(np.full(3, a), proposal_b)
        init = np.zeros((DISTRIBUTION_SAMPLE_SIZE, 3))

        burn_in = sample_coupled_batches(
            init, target, proposal, variance, BURN_IN, 7, seed=912_301
        )
        sample = sample_coupled_batches(
            burn_in.trajectories[:, -1, :],
            target,
            proposal,
            variance,
            1,
            7,
            seed=912_302,
        ).trajectories[:, -1, :]

        analytic_mean = target_b / (1.0 - a)
        analytic_covariance = np.eye(3) * variance / (1.0 - a**2)
        sample_mean = np.mean(sample, axis=0)
        sample_covariance = np.cov(sample, rowvar=False, bias=False)

        np.testing.assert_array_less(
            np.linalg.norm(sample_mean - analytic_mean), MEAN_TOLERANCE
        )
        np.testing.assert_array_less(
            np.linalg.norm(sample_covariance - analytic_covariance, ord="fro"),
            COVARIANCE_TOLERANCE,
        )

        directions = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 0.0],
                [1.0, 2.0, -1.0],
            ]
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        projected_std = math.sqrt(
            variance / (1.0 - a**2)
        )
        for direction in directions:
            projected = sample @ direction
            location = analytic_mean @ direction
            for probability in TEST_CDF_PROBABILITIES:
                threshold = location + projected_std * standard_normal_quantile(probability)
                empirical_cdf = float(np.mean(projected <= threshold))
                self.assertLessEqual(
                    abs(empirical_cdf - probability),
                    CDF_TOLERANCE,
                    msg=(direction, probability, empirical_cdf),
                )

    def test_target_baseline_stationary_mean_and_covariance(self):
        a = 0.72
        variance = 0.25
        b = np.array([0.15, -0.20, 0.30])
        target = AffineMean(np.full(3, a), b)
        burned = sample_target_baseline(
            np.zeros((4_000, 3)), target, variance, BURN_IN, seed=8001
        )[:, -1, :]
        sample = sample_target_baseline(burned, target, variance, 1, seed=8002)[:, -1, :]
        analytic_mean = b / (1.0 - a)
        analytic_covariance = np.eye(3) * variance / (1.0 - a**2)
        np.testing.assert_array_less(np.linalg.norm(sample.mean(0) - analytic_mean), 0.08)
        np.testing.assert_array_less(
            np.linalg.norm(np.cov(sample, rowvar=False) - analytic_covariance, ord="fro"),
            0.20,
        )


if __name__ == "__main__":
    unittest.main()
