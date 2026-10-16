"""Small accounting and timing checks for the strategy engine."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from strategy import momentum_positions, select_winners, summary, trading_returns
from study import LOOKBACKS, evaluate, stationary_returns


class StrategyTests(unittest.TestCase):
    def test_positions_use_only_previous_returns(self):
        returns = np.array([[.01, -.02, .03, -.04, .05, .06, -.07, .08]])
        positions = momentum_positions(returns, np.array([1, 3]), start=3)
        expected = [[[1, -1, 1, 1, -1], [1, -1, 1, 1, 1]]]
        assert_array_equal(positions, expected)
        self.assertEqual(positions.shape, (1, 2, 5))
        net, _ = trading_returns(returns[:, 3:], positions)
        assert_allclose(net, positions * returns[:, None, 3:])

    def test_current_return_changes_next_position_not_current(self):
        returns = np.array([[.02, -.03, .04, .01, -.02, .05]])
        changed = returns.copy()
        changed[0, 3] = -.01
        before = momentum_positions(returns, np.array([1]), start=2)
        after = momentum_positions(changed, np.array([1]), start=2)
        assert_array_equal(before[..., :2], after[..., :2])
        self.assertEqual(before[0, 0, 2], 1)
        self.assertEqual(after[0, 0, 2], -1)

    def test_prefix_and_future_perturbations_leave_past_unchanged(self):
        returns = np.random.default_rng(12).normal(0, .01, (2, 20))
        lookbacks = np.array([1, 3, 5])
        original = momentum_positions(returns, lookbacks, start=5)
        prefix = momentum_positions(returns[:, :12], lookbacks, start=5)
        changed = returns.copy()
        changed[:, 12:] += 1
        perturbed = momentum_positions(changed, lookbacks, start=5)
        assert_array_equal(original[..., :7], prefix)
        assert_array_equal(original[..., :7], perturbed[..., :7])

    def test_zero_signal_means_flat_position(self):
        positions = momentum_positions(
            np.array([[.1, -.1, 0., .2]]), np.array([2]), start=2
        )
        assert_array_equal(positions, [[[0, -1]]])

    def test_ledger_charges_entry_flip_exit_and_final_liquidation(self):
        returns = np.array([[.01, -.02, .03, .04, -.01]])
        positions = np.array([[[1, 1, -1, 0, 1]]])
        net, turnover = trading_returns(returns, positions, cost_bps=10)
        assert_array_equal(turnover, [[[1, 0, 2, 1, 2]]])
        assert_allclose(net, [[[.009, -.020, -.032, -.001, -.012]]])
        self.assertAlmostEqual(float(net.sum()), -.056)

    def test_costs_reduce_returns_without_changing_turnover(self):
        returns = np.array([[.01, .02, -.03, .04]])
        positions = np.array([[[1, 1, -1, -1], [0, 1, 0, -1]]])
        gross, turnover = trading_returns(returns, positions, cost_bps=0)
        net, same_turnover = trading_returns(returns, positions, cost_bps=5)
        assert_array_equal(turnover, same_turnover)
        assert_allclose(gross - net, .0005 * turnover)
        self.assertTrue(np.all(net <= gross))

    def test_summary_uses_arithmetic_mean_and_sample_standard_deviation(self):
        result = summary(np.array([[[.01, -.02, .03, 0.]]]))
        assert_allclose(result["mean_daily"], [[.005]])
        assert_allclose(result["annual_mean"], [[1.26]])
        expected_sharpe = np.sqrt(252) * .005 / np.sqrt(.0013 / 3)
        assert_allclose(result["sharpe"], [[expected_sharpe]])

    def test_zero_variance_has_no_valid_sharpe(self):
        result = summary(np.full((2, 3, 4), .01))
        self.assertTrue(np.isnan(result["sharpe"]).all())
        assert_allclose(result["annual_mean"], 2.52)

    def test_duplicate_rules_do_not_create_a_different_winner(self):
        scores = np.array([[.2, .5, .1, .4, .3], [.6, .4, .2, .1, 0.]])
        original = select_winners(scores, np.arange(5))
        duplicated = select_winners(scores, np.repeat(np.arange(5), 10))
        assert_array_equal(original, [1, 0])
        assert_array_equal(original, duplicated)
        pnl = np.arange(40, dtype=float).reshape(2, 5, 4) / 1000
        rows = np.arange(2)
        assert_array_equal(pnl[rows, original], pnl[rows, duplicated])

    def test_nested_search_cannot_lower_best_training_score(self):
        scores = np.array([[.1, .8, -.2, .9], [.7, -.1, .9, .2]])
        rows = np.arange(2)
        small = select_winners(scores, np.array([0, 1]))
        large = select_winners(scores, np.array([0, 1, 2, 3]))
        self.assertTrue(np.all(scores[rows, large] >= scores[rows, small]))

    def test_selection_ignores_invalid_scores_and_breaks_ties_by_order(self):
        scores = np.array([[np.nan, 2., 2.], [1., np.inf, 3.]])
        assert_array_equal(select_winners(scores, np.array([2, 0, 1])), [2, 2])
        with self.assertRaises(ValueError):
            select_winners(np.array([[np.nan, np.inf]]), np.array([0, 1]))

    def test_invalid_inputs_fail_before_calculation(self):
        returns = np.ones((1, 5))
        for lookbacks in (np.array([0]), np.array([1, 1]), np.array([1.5])):
            with self.subTest(lookbacks=lookbacks), self.assertRaises(ValueError):
                momentum_positions(returns, lookbacks, start=2)
        for start in (0, 5):
            with self.subTest(start=start), self.assertRaises(ValueError):
                momentum_positions(returns, np.array([1]), start=start)
        with self.assertRaises(ValueError):
            momentum_positions(np.array([[0., np.nan, 0.]]), np.array([1]), 1)
        with self.assertRaises(ValueError):
            trading_returns(returns, np.ones((1, 2, 4)))
        with self.assertRaises(ValueError):
            trading_returns(returns, np.ones((1, 2, 5)), cost_bps=-1)
        with self.assertRaises(ValueError):
            select_winners(np.ones((1, 2)), np.array([2]))


class WeakSignalTests(unittest.TestCase):
    def test_stationary_generator_scales_innovations(self):
        shocks = np.array([[.01, -.02, .03], [-.02, .01, 0.]])
        result = stationary_returns(shocks, rho=.6)
        assert_allclose(result, [[.01, -.01, .018], [-.02, -.004, -.0024]])

    def test_zero_signal_preserves_iid_sample_without_mutating_input(self):
        shocks = np.random.default_rng(14).normal(0, .01, (2, 12))
        original = shocks.copy()
        result = stationary_returns(shocks, rho=0)
        assert_array_equal(result, original)
        self.assertFalse(np.shares_memory(result, shocks))
        stationary_returns(shocks, rho=.15)
        assert_array_equal(shocks, original)

    def test_future_shocks_cannot_change_past_returns(self):
        shocks = np.random.default_rng(15).normal(0, .01, (2, 20))
        changed = shocks.copy()
        changed[:, 10:] += .03
        result = stationary_returns(shocks, rho=.15)
        perturbed = stationary_returns(changed, rho=.15)
        prefix = stationary_returns(shocks[:, :10], rho=.15)
        assert_array_equal(result[:, :10], perturbed[:, :10])
        assert_array_equal(result[:, :10], prefix)

    def test_generator_rejects_invalid_shapes_and_parameters(self):
        for shocks in (np.ones(3), np.ones((1, 2, 3)), np.empty((0, 2)),
                       np.empty((1, 0)), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.subTest(shocks=shocks), self.assertRaises(ValueError):
                stationary_returns(shocks, rho=.1)
        for rho in (-1, 1, 1.1, np.nan, np.inf):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                stationary_returns(np.ones((1, 3)), rho=rho)

    def test_serial_dependence_keeps_selection_before_test_and_duplicates_equal(self):
        shocks = np.random.default_rng(16).normal(0, .01, (3, 332))
        returns = stationary_returns(shocks, rho=.15)
        changed = returns.copy()
        changed[:, 292:] *= -2

        def score(sample):
            positions = momentum_positions(sample, LOOKBACKS, start=252)
            return evaluate(sample[:, 252:292], sample[:, 292:],
                            positions[:, :, :40], positions[:, :, 40:])

        original, perturbed = score(returns), score(changed)
        for label in original:
            assert_array_equal(original[label]["lookback"], perturbed[label]["lookback"])
            assert_array_equal(original[label]["train_sharpe"],
                               perturbed[label]["train_sharpe"])
        for output in (original, perturbed):
            for metric in output["5 rules"]:
                assert_array_equal(output["5 rules"][metric],
                                   output["5 rules repeated 10 times"][metric])


if __name__ == "__main__":
    unittest.main()
