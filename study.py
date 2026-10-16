"""Compare parameter searches under no signal and weak serial dependence."""

import argparse
import csv
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from strategy import momentum_positions, select_winners, summary, trading_returns

ROOT = Path(__file__).resolve().parent
LOOKBACKS = np.array([
    2, 10, 40, 100, 252, 58, 3, 13, 54, 38, 14, 23, 219, 71, 8, 41, 101,
    16, 117, 94, 47, 18, 144, 125, 50, 11, 6, 27, 190, 154, 5, 19, 62,
    134, 33, 31, 7, 88, 35, 77, 109, 15, 235, 22, 44, 204, 178, 9, 25, 20,
])
GROUPS = {
    "5 rules": np.arange(5),
    "20 rules": np.arange(20),
    "50 rules": np.arange(50),
    "5 rules repeated 10 times": np.repeat(np.arange(5), 10),
}
SEED, PATHS, WARMUP, TRAIN, TEST = 2026101607, 1000, 252, 756, 756
COSTS = (0, 1, 5)
RHOS = (0.0, 0.05, 0.10)
DATA_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "5_Industry_Portfolios_daily_CSV.zip"
)


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("intervals require at least two finite independent-path values")
    mean = values.mean()
    half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return {"mean": float(mean), "low": float(mean - half_width),
            "high": float(mean + half_width)}


def evaluate(train_returns, test_returns, train_positions, test_positions):
    train_pnl, _ = trading_returns(train_returns, train_positions)
    test_pnl, test_turnover = trading_returns(test_returns, test_positions)
    train_stats, test_stats = summary(train_pnl), summary(test_pnl)
    rows = np.arange(train_returns.shape[0])
    output = {}
    for label, candidates in GROUPS.items():
        winner = select_winners(train_stats["sharpe"], candidates)
        annual_turnover = 252 * test_turnover[rows, winner].mean(axis=1)
        annual_gross = test_stats["annual_mean"][rows, winner]
        output[label] = {
            "lookback": LOOKBACKS[winner],
            "train_sharpe": train_stats["sharpe"][rows, winner],
            "test_sharpe": test_stats["sharpe"][rows, winner],
            "test_gross_annual": annual_gross,
            "annual_turnover": annual_turnover,
            # Winners are fixed before costs are varied.
            **{f"test_net_{cost}bps": annual_gross - cost / 10000 * annual_turnover
               for cost in COSTS},
        }
    for key in output["5 rules"]:
        if not np.array_equal(output["5 rules"][key],
                              output["5 rules repeated 10 times"][key]):
            raise AssertionError("duplicate candidates changed the result")
    return output


def stationary_returns(shocks, rho):
    """Keep marginal volatility fixed while adding AR(1) dependence."""
    shocks = np.asarray(shocks, dtype=float)
    if shocks.ndim != 2 or 0 in shocks.shape or not np.isfinite(shocks).all():
        raise ValueError("shocks must be a nonempty finite 2D array")
    if not np.isscalar(rho) or isinstance(rho, (str, bytes, bool, np.bool_)):
        raise ValueError("rho must be a finite scalar with absolute value below one")
    try:
        rho = float(rho)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid rho") from error
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must have absolute value below one")
    returns = shocks.copy()
    # The first draw has the stationary marginal variance, rather than starting at zero.
    if rho != 0:
        scale = np.sqrt(1 - rho**2)
        for day in range(1, shocks.shape[1]):
            returns[:, day] = rho * returns[:, day - 1] + scale * shocks[:, day]
    return returns


def simulate(rho=0.0):
    # Resetting the same seed pairs the shocks across dependence levels.
    rng = np.random.default_rng(SEED)
    collected = {label: {} for label in GROUPS}
    for begin in range(0, PATHS, 100):
        count = min(100, PATHS - begin)
        shocks = rng.normal(0, .01, size=(count, WARMUP + TRAIN + TEST))
        returns = stationary_returns(shocks, rho)
        positions = momentum_positions(returns, LOOKBACKS, WARMUP)
        scored = returns[:, WARMUP:]
        batch = evaluate(scored[:, :TRAIN], scored[:, TRAIN:],
                         positions[:, :, :TRAIN], positions[:, :, TRAIN:])
        for label, metrics in batch.items():
            for name, values in metrics.items():
                collected[label].setdefault(name, []).append(values)
    return {label: {name: np.concatenate(chunks) for name, chunks in metrics.items()}
            for label, metrics in collected.items()}


def aggregate(paths):
    return {
        "groups": {
            label: {name: mean_interval(values) for name, values in metrics.items()
                    if name != "lookback"}
            for label, metrics in paths.items()
        },
        "paired_50_minus_5": {
            name: mean_interval(paths["50 rules"][name] - paths["5 rules"][name])
            for name in ("train_sharpe", "test_sharpe", "test_gross_annual",
                         "test_net_5bps", "annual_turnover")
        },
        "shortest_window_selection_rate": {
            label: float(np.mean(metrics["lookback"] == 2))
            for label, metrics in paths.items()
        },
        "duplicate_control_exact": True,
    }


def download_data(cache):
    target = cache / "5_industry_daily.zip"
    if not target.exists():
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Python research study"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if archive.testzip() is not None:
                raise ValueError("downloaded archive is corrupt")
        target.write_bytes(raw)
        metadata = {"url": DATA_URL,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "sha256": hashlib.sha256(raw).hexdigest()}
        (cache / "download.json").write_text(json.dumps(metadata, indent=2))
    raw = target.read_bytes()
    metadata = json.loads((cache / "download.json").read_text())
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("cached data differ from their recorded hash")
    return raw, metadata


def read_consumer_returns(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("expected one CSV in the official archive")
        lines = archive.read(names[0]).decode("utf-8-sig").splitlines()
    marker = next(i for i, line in enumerate(lines)
                  if "average value weighted returns" in line.lower())
    header = next(i for i in range(marker + 1, len(lines)) if "Cnsmr" in lines[i])
    column = [value.strip() for value in next(csv.reader([lines[header]]))].index("Cnsmr")
    dates, values = [], []
    for line in lines[header + 1:]:
        fields = next(csv.reader([line]))
        if not fields or not fields[0].strip().isdigit():
            break
        date = fields[0].strip()
        datetime.strptime(date, "%Y%m%d")
        if "19980101" <= date <= "20251231":
            value = float(fields[column])
            if not np.isfinite(value) or value in (-99.99, -999):
                raise ValueError(f"missing consumer return at {date}")
            dates.append(date)
            values.append(value / 100)
    if len(dates) <= WARMUP or dates != sorted(set(dates)):
        raise ValueError("historical dates must be unique, sorted and long enough")
    return np.array(dates), np.array(values)


def historical_example(cache):
    raw, metadata = download_data(cache)
    dates, returns = read_consumer_returns(raw)
    positions = momentum_positions(returns[None, :], LOOKBACKS, WARMUP)
    dates, scored = dates[WARMUP:], returns[None, WARMUP:]
    train = (dates >= "20000101") & (dates <= "20151231")
    test = (dates >= "20160101") & (dates <= "20251231")
    if train.sum() < 3000 or test.sum() < 2000:
        raise ValueError("the frozen historical periods are incomplete")
    groups = evaluate(scored[:, train], scored[:, test],
                      positions[:, :, train], positions[:, :, test])
    passive, _ = trading_returns(scored[:, test], np.ones((1, 1, int(test.sum()))))
    return {
        "source": metadata, "series": "Value-weighted Cnsmr industry research portfolio",
        "train_period": [str(dates[train][0]), str(dates[train][-1])],
        "test_period": [str(dates[test][0]), str(dates[test][-1])],
        "train_days": int(train.sum()), "test_days": int(test.sum()),
        "groups": {label: {name: float(values[0]) for name, values in metrics.items()}
                   for label, metrics in groups.items()},
        "passive_gross_annual": float(summary(passive)["annual_mean"][0, 0]),
        "limitations": "Revised historical snapshot; illustrative split, not live validation. "
                       "Industry research portfolio is not directly tradeable. Costs are hypothetical; "
                       "borrow fees, financing and risk-free returns are not modelled.",
    }


def make_figure(results, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), layout="constrained")
    blue, red, green = "#285A8E", "#B4513D", "#357468"
    labels = ["5 rules", "20 rules", "50 rules"]
    scenarios = [results["synthetic"], *(results["weak_signal"][f"{rho:.2f}"]
                                       for rho in RHOS[1:])]

    def plot_intervals(axis, stats, color, label):
        means = np.array([100 * value["mean"] for value in stats])
        errors = [[100 * (v["mean"] - v["low"]) for v in stats],
                  [100 * (v["high"] - v["mean"]) for v in stats]]
        axis.errorbar(RHOS, means, yerr=errors, fmt="o-", color=color,
                      capsize=3, label=label)

    for label, color in zip(labels, (blue, green, red)):
        plot_intervals(axes[0], [s["groups"][label]["test_net_5bps"] for s in scenarios],
                       color, label)
    axes[0].set(ylabel="Mean test P&L (% per year, arithmetic)",
                title="1. Test results after 5 bps costs")
    axes[0].axhline(0, color="#888888", lw=.7)
    axes[0].legend(frameon=False)

    for field, color, label in (("test_gross_annual", blue, "Before costs"),
                                 ("test_net_5bps", red, "After 5 bps costs")):
        plot_intervals(axes[1], [s["paired_50_minus_5"][field] for s in scenarios],
                       color, label)
    axes[1].set(ylabel="50 minus 5: test P&L (percentage points/year)",
                title="2. Does a denser search help?")
    axes[1].axhline(0, color="#888888", lw=.7)
    axes[1].legend(frameon=False)

    for label, color in zip(labels, (blue, green, red)):
        rates = [100 * s["shortest_window_selection_rate"][label] for s in scenarios]
        axes[2].plot(RHOS, rates, "o-", color=color, label=label)
    axes[2].set(ylabel="Paths selecting the 2-day rule (%)", ylim=(0, 100),
                title="3. Which rule wins selection?")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set(xticks=RHOS, xticklabels=["0", "0.05", "0.10"],
                 xlabel="Preset return dependence (rho)")
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("Does searching more settings help when a weak signal exists?",
                 fontsize=14, fontweight="bold")
    fig.supxlabel(
        f"{PATHS:,} independent paths, shared shocks across settings | "
        "Bars: pointwise 95% Monte Carlo intervals for means | "
        "Gross-score selection; fixed winners under costs",
        fontsize=9,
    )
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", action="store_true",
                        help="also download/cache the fixed Ken French illustration")
    args = parser.parse_args()
    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    paths_by_rho = {rho: simulate(rho) for rho in RHOS}
    scenarios = {rho: aggregate(paths) for rho, paths in paths_by_rho.items()}
    results = {
        "design": {"seed": SEED, "paths": PATHS, "warmup_days": WARMUP,
                   "selection_days": TRAIN, "test_days": TEST,
                   "daily_return_distribution": "Stationary Gaussian AR(1); marginal sd 0.01",
                   "rho_values": list(RHOS),
                   "initialization": "r[0]=shock[0]; shocks iid Normal(0, 0.01^2)",
                   "recurrence": "r[t]=rho*r[t-1]+sqrt(1-rho^2)*shock[t]",
                   "pairing": "Shared shocks across rho and rules; 1000 independent whole paths",
                   "holdout": "Continuous forward holdout; train and test segments are dependent",
                   "primary_comparison": "50 minus 5 test net arithmetic P&L at 5 bps, per rho",
                   "lookbacks": LOOKBACKS.tolist(), "cost_bps": list(COSTS),
                   "selection": "Training gross Sharpe; fixed across test cost scenarios",
                   "score": "sqrt(252) * mean / sample sd; descriptive, not dependence-adjusted annual Sharpe",
                   "pnl": "Fixed-notional arithmetic P&L; no compounding, financing or borrow fees",
                   "intervals": "Pointwise mean +/- 1.96 * sd / sqrt(paths); not simultaneous or model-risk bounds",
                   "formal_vs_pilot": "Original null used a new seed; weak extension reuses it after fixing design"},
        "synthetic": scenarios[0.0],
        "weak_signal": {f"{rho:.2f}": scenarios[rho] for rho in RHOS[1:]},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_hashes": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in ("strategy.py", "study.py")},
    }
    if args.historical:
        results["historical"] = historical_example(cache)
    with (cache / "path_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        fields = list(paths_by_rho[0.0]["5 rules"])
        writer.writerow(["rho", "path", "group", *fields])
        for rho, paths in paths_by_rho.items():
            for label, metrics in paths.items():
                for i in range(PATHS):
                    writer.writerow([rho, i, label, *(metrics[field][i] for field in fields)])
    make_figure(results, ROOT / "results.png")
    (ROOT / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("Saved results.json and results.png")
    for rho, scenario in scenarios.items():
        print(f"rho={rho:.2f}")
        for label in ("5 rules", "20 rules", "50 rules"):
            group = scenario["groups"][label]
            rate = scenario["shortest_window_selection_rate"][label]
            print(f"  {label}: selection score {group['train_sharpe']['mean']:.3f}; "
                  f"net at 5 bps {100*group['test_net_5bps']['mean']:.2f}%/year; "
                  f"2-day selected {100*rate:.1f}%")
        delta = scenario["paired_50_minus_5"]["test_net_5bps"]
        print(f"  50 minus 5 net: {100*delta['mean']:.3f} pp/year; "
              f"95% interval [{100*delta['low']:.3f}, {100*delta['high']:.3f}]")
    print("Duplicate-rule control: identical on every path, at every rho")



if __name__ == "__main__":
    main()

