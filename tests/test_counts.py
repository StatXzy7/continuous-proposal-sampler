"""Tail-batch submission and the two distinct counters (states vs batches)."""

import os
import sys

import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gauss_batch import sample_coupled_batches


def kernels(d=2, s2=0.5):
    return ((0.8 * np.eye(d), np.full(d, 0.1), s2),
            (0.5 * np.eye(d), np.zeros(d), s2))


class TestCountsAndTailBatch(unittest.TestCase):
    def test_all_accepted_counts_and_no_over_submission(self):
        target, proposal = kernels()
        x0 = np.zeros(2)
        # identical mean function => every draft accepted
        res = sample_coupled_batches(x0, target, target, 5, 2,
                                     n_traj=2, seed=3)
        self.assertEqual(res.trajectories.shape, (2, 6, 2))  # never more than T+1
        self.assertEqual(res.n_target_evaluations, 2 * 5)
        self.assertEqual(res.n_batches, 2 * 3)  # batches: 2 + 2 + 1
        np.testing.assert_array_equal(res.target_evaluations_per_traj, [5, 5])
        np.testing.assert_array_equal(res.batches_per_traj, [3, 3])
        # last batch accepted only one state despite lookahead 2
        for runs in res.accept_run_lengths:
            self.assertEqual(sum(runs), 5)
            self.assertEqual(runs, [2, 2, 1])

    def test_tail_sizes_for_various_lookaheads(self):
        target, _ = kernels()
        x0 = np.zeros(2)
        for T, L, expected_batch_sizes in [
            (7, 3, [3, 3, 1]),
            (6, 3, [3, 3]),
            (1, 5, [1]),
            (4, 4, [4]),
        ]:
            res = sample_coupled_batches(x0, target, target, T, L,
                                         n_traj=1, seed=0)
            self.assertEqual(res.n_batches, len(expected_batch_sizes))
            self.assertEqual(res.accept_run_lengths[0], expected_batch_sizes)
            self.assertEqual(res.trajectories.shape[1], T + 1)

    def test_rejection_inflates_state_counter_not_min_steps(self):
        target, proposal = kernels()
        x0 = np.zeros(2)
        res = sample_coupled_batches(x0, target, proposal, 4, 3,
                                     n_traj=40, seed=11)
        # counters are summed consistently
        self.assertEqual(res.n_target_evaluations,
                         int(res.target_evaluations_per_traj.sum()))
        self.assertEqual(res.n_batches, int(res.batches_per_traj.sum()))
        # baseline work without rejections would be 4 evals / 2 batches;
        # rejected drafts waste evaluations but trajectories still have T steps
        self.assertEqual(res.trajectories.shape, (40, 5, 2))
        # whenever a trajectory was rejected, it paid extra evaluations
        for i, rej in enumerate(res.rejection_steps):
            if rej >= 0:
                self.assertGreaterEqual(res.target_evaluations_per_traj[i], 4)
                self.assertGreaterEqual(res.batches_per_traj[i], 2)
        # the two counters measure different things
        self.assertNotEqual(res.n_target_evaluations, res.n_batches)

    def test_first_rejection_recorded_only(self):
        target, proposal = kernels()
        x0 = np.zeros(2)
        # high rejection pressure
        tp = (np.eye(2), np.array([3.0, -3.0]), 0.05)
        pp = (np.eye(2), np.zeros(2), 0.05)
        res = sample_coupled_batches(x0, tp, pp, 10, 4, n_traj=30, seed=4)
        self.assertGreater(int((res.rejection_steps >= 0).sum()), 20)
        for i, rej in enumerate(res.rejection_steps):
            if rej >= 0:
                self.assertGreaterEqual(rej, 0)
                self.assertLess(int(rej), 10)
        self.assertGreaterEqual(res.n_rejections,
                                int((res.rejection_steps >= 0).sum()))


if __name__ == "__main__":
    unittest.main()
