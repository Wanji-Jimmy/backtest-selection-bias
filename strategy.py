"""Lagged momentum rules, trading costs and training-only selection."""

import numpy as np


def _finite_array(values, ndim, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty {ndim}D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty 1D integer array")
    return array


def momentum_positions(returns, lookbacks, start):
    """Return positions for days start onward, using only earlier returns."""
    returns = _finite_array(returns, 2, "returns")
    lookbacks = _integer_vector(lookbacks, "lookbacks")
    if np.any(lookbacks <= 0) or np.unique(lookbacks).size != lookbacks.size:
        raise ValueError("lookbacks must be positive and unique")
    if isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer)):
        raise ValueError("start must be an integer")
    if start < lookbacks.max() or start >= returns.shape[1]:
        raise ValueError("start must allow every lookback and be smaller than the day count")

    prefix = np.concatenate(
        (np.zeros((returns.shape[0], 1)), np.cumsum(returns, axis=1)), axis=1
    )
    days = np.arange(start, returns.shape[1])
    # prefix[t] ends at t - 1, so the current day's return cannot affect its position.
    history = prefix[:, days][:, None, :] - prefix[
        :, days[None, :] - lookbacks.astype(np.intp)[:, None]
    ]
    return np.sign(history)


def trading_returns(returns, positions, cost_bps=0.0):
    """Return net daily returns and turnover, starting and ending flat."""
    returns = _finite_array(returns, 2, "returns")
    positions = _finite_array(positions, 3, "positions")
    if positions.shape[0] != returns.shape[0] or positions.shape[2] != returns.shape[1]:
        raise ValueError("positions must match the paths and days in returns")
    if not np.isscalar(cost_bps) or isinstance(cost_bps, (str, bytes, bool, np.bool_)):
        raise ValueError("cost_bps must be a finite nonnegative scalar")
    try:
        cost_bps = float(cost_bps)
    except (TypeError, ValueError) as error:
        raise ValueError("cost_bps must be a finite nonnegative scalar") from error
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be a finite nonnegative scalar")

    turnover = np.empty_like(positions)
    turnover[:, :, 0] = np.abs(positions[:, :, 0])
    turnover[:, :, 1:] = np.abs(np.diff(positions, axis=2))
    turnover[:, :, -1] += np.abs(positions[:, :, -1])
    net_returns = positions * returns[:, None, :] - cost_bps / 10_000 * turnover
    return net_returns, turnover


def summary(net_returns):
    """Summarise each path and rule; constant returns have an undefined Sharpe."""
    net_returns = _finite_array(net_returns, 3, "net_returns")
    if net_returns.shape[2] < 2:
        raise ValueError("summary requires at least two days")
    mean_daily = np.mean(net_returns, axis=2)
    sample_sd = np.std(net_returns, axis=2, ddof=1)
    sharpe = np.full(mean_daily.shape, np.nan)
    np.divide(
        np.sqrt(252) * mean_daily,
        sample_sd,
        out=sharpe,
        where=sample_sd > 1e-12,
    )
    return {
        "mean_daily": mean_daily,
        "annual_mean": 252 * mean_daily,
        "sharpe": sharpe,
    }


def select_winners(scores, candidate_indices):
    """Select global rule indices, resolving ties by candidate order."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError("scores must be a nonempty 2D array")
    candidate_indices = _integer_vector(candidate_indices, "candidate_indices")
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= scores.shape[1]):
        raise ValueError("candidate_indices must refer to existing rules")
    candidate_indices = candidate_indices.astype(np.intp)
    candidates = scores[:, candidate_indices]
    valid = np.isfinite(candidates)
    if not valid.any(axis=1).all():
        raise ValueError("each path must have at least one finite candidate score")
    best = np.argmax(np.where(valid, candidates, -np.inf), axis=1)
    return candidate_indices[best]
