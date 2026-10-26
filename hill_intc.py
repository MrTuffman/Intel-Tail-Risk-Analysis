"""Hill estimator on Intel (INTC) losses + Gaussian 99% VaR breaches."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TICKER = "INTC"
NAME = "Intel"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"
PRICE_CSV = DATA_DIR / "HistoricalData_1786949137896.csv"


def load_closes(path: Path) -> pd.Series:
    """Load daily closes from the Nasdaq-style CSV in data/."""
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    closes = (
        df["Close/Last"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .astype(float)
    )
    series = pd.Series(closes.to_numpy(), index=dates, name="close").dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def log_returns(prices: pd.Series) -> pd.Series:
    """Convert prices to daily log returns."""
    return np.log(prices / prices.shift(1)).dropna().rename("log_return")


def hill_alpha(losses: np.ndarray, k: int) -> float:
    """Hill estimator for a single k."""
    x = np.sort(losses)[::-1]
    if k < 1 or k >= len(x):
        raise ValueError("k must satisfy 1 <= k < n")
    return float(1.0 / (np.log(x[:k]).mean() - np.log(x[k])))


def hill_alphas(losses: np.ndarray, kmax: int) -> np.ndarray:
    """Vectorized Hill estimates for k = 1..kmax."""
    lx = np.log(np.sort(losses)[::-1])
    k = np.arange(1, kmax + 1)
    return 1.0 / (np.cumsum(lx)[:kmax] / k - lx[k])


def choose_k(alphas: np.ndarray, k_min: int = 20, window_frac: float = 0.15) -> int:
    """Pick k in the flattest region of the Hill plot."""
    kmax = len(alphas)
    window = max(15, int(window_frac * kmax))
    if k_min + window >= kmax:
        return max(k_min, kmax // 4)

    best_k = k_min
    best_score = np.inf
    for start in range(k_min - 1, kmax - window):
        chunk = alphas[start : start + window]
        mean = chunk.mean()
        if mean <= 0:
            continue
        score = chunk.std(ddof=1) / mean
        if score < best_score:
            best_score = score
            best_k = start + window // 2 + 1
    return int(best_k)


def gaussian_var_breaches(returns: pd.Series, level: float = 0.99) -> dict:
    """Count breaches of a constant Gaussian VaR on log returns."""
    from scipy.stats import norm

    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    var = float(norm.ppf(1.0 - level, loc=mu, scale=sigma))
    breaches = returns < var
    n_years = (returns.index[-1] - returns.index[0]).days / 365.25
    expected = (1.0 - level) * len(returns)
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "var": float(var),
        "n_breaches": int(breaches.sum()),
        "breach_dates": returns.index[breaches],
        "breach_returns": returns[breaches],
        "expected_breaches": float(expected),
        "expected_per_year": float(expected / n_years) if n_years > 0 else np.nan,
        "actual_per_year": float(breaches.sum() / n_years) if n_years > 0 else np.nan,
        "n_years": float(n_years),
    }


def plot_hill(alphas: np.ndarray, k_star: int, alpha_star: float, path: Path) -> None:
    """Plot Hill alpha vs k and mark the chosen threshold."""
    k = np.arange(1, len(alphas) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(k, alphas, color="#1f4e79", lw=1.4, label=r"$\hat{\alpha}_k$")
    ax.axvline(k_star, color="#c45c26", ls="--", lw=1.2, label=f"chosen k = {k_star}")
    ax.axhline(alpha_star, color="#c45c26", ls=":", lw=1.0, label=rf"$\hat{{\alpha}}$ = {alpha_star:.2f}")
    ax.set_xlabel("order statistics k (largest losses)")
    ax.set_ylabel(r"Hill estimate $\hat{\alpha}_k$")
    ax.set_title(f"{NAME} ({TICKER}): Hill plot of left-tail exponent")
    ax.set_ylim(0, min(12, np.nanpercentile(alphas[20:], 99) * 1.3 if len(alphas) > 20 else 12))
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tail_loglog(losses: np.ndarray, k_star: int, alpha_star: float, path: Path) -> None:
    """Plot empirical survival against the fitted power-law tail."""
    x = np.sort(losses)[::-1]
    n = len(x)
    ranks = np.arange(1, n + 1)
    emp = ranks / (n + 1.0)
    thresh = x[k_star]
    xx = np.linspace(thresh, x[0], 200)
    c = (k_star / (n + 1.0)) * (thresh ** alpha_star)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.loglog(x, emp, ".", ms=3.5, color="#1f4e79", alpha=0.7, label="empirical survival")
    ax.loglog(xx, c * xx ** (-alpha_star), color="#c45c26", lw=2.0, label=rf"power law $\alpha$={alpha_star:.2f}")
    ax.axvline(thresh, color="gray", ls="--", lw=1.0, label=f"threshold X_(k+1), k={k_star}")
    ax.set_xlabel("loss size (−log return)")
    ax.set_ylabel("P(X > x)")
    ax.set_title(f"{NAME} ({TICKER}): left-tail survival vs power-law fit")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_returns_vs_normal(returns: pd.Series, path: Path) -> None:
    """Compare the return density to a Gaussian fit."""
    r = returns.values
    xs = np.linspace(r.min(), r.max(), 400)
    mu, sigma = r.mean(), r.std(ddof=1)
    normal_pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(r, bins=60, density=True, color="#1f4e79", alpha=0.55, label=f"{TICKER} log returns")
    ax.plot(xs, normal_pdf, color="#c45c26", lw=2.0, label="Gaussian fit")
    ax.set_xlabel("daily log return")
    ax.set_ylabel("density")
    ax.set_title(f"{NAME} ({TICKER}): return density vs Gaussian")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_var_breaches(returns: pd.Series, var_info: dict, path: Path) -> None:
    """Plot returns and mark Gaussian VaR breach days."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(returns.index, returns.values, color="#1f4e79", lw=0.8, label="daily log return")
    ax.axhline(var_info["var"], color="#c45c26", ls="--", lw=1.3, label=f"Gaussian 99% VaR = {var_info['var']:.3%}")
    ax.scatter(
        var_info["breach_dates"],
        var_info["breach_returns"],
        color="#c45c26",
        s=28,
        zorder=3,
        label=f"breaches ({var_info['n_breaches']})",
    )
    ax.set_title(f"{NAME} ({TICKER}): Gaussian 99% VaR breaches")
    ax.set_ylabel("log return")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Run the INTC Hill + VaR pipeline and write outputs."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_closes(PRICE_CSV)
    returns = log_returns(prices)
    losses = (-returns).clip(lower=0)
    # Positive loss magnitudes only (down days)
    loss_vals = losses[losses > 0].to_numpy()

    prices.to_csv(DATA_DIR / f"{TICKER.lower()}_close.csv", header=True)
    returns.to_csv(DATA_DIR / f"{TICKER.lower()}_log_returns.csv", header=True)

    n = len(loss_vals)
    kmax = max(30, min(n // 4, 200))
    alphas = hill_alphas(loss_vals, kmax)
    k_star = choose_k(alphas, k_min=max(15, n // 40))
    alpha_star = float(alphas[k_star - 1])

    var_info = gaussian_var_breaches(returns, level=0.99)

    plot_hill(alphas, k_star, alpha_star, FIG_DIR / "hill_plot.png")
    plot_tail_loglog(loss_vals, k_star, alpha_star, FIG_DIR / "tail_loglog.png")
    plot_returns_vs_normal(returns, FIG_DIR / "returns_vs_normal.png")
    plot_var_breaches(returns, var_info, FIG_DIR / "var_breaches.png")

    # Moments exist only for order p < alpha under a power-law tail
    kurtosis_finite = alpha_star > 4
    variance_finite = alpha_star > 2

    summary = {
        "ticker": TICKER,
        "name": NAME,
        "source_csv": PRICE_CSV.name,
        "n_prices": int(len(prices)),
        "n_returns": int(len(returns)),
        "n_positive_losses": int(n),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "k_star": int(k_star),
        "alpha_hat": alpha_star,
        "threshold_loss": float(np.sort(loss_vals)[::-1][k_star]),
        "variance_finite": bool(variance_finite),
        "kurtosis_finite": bool(kurtosis_finite),
        "gaussian_99_var": var_info["var"],
        "n_breaches": var_info["n_breaches"],
        "expected_breaches": var_info["expected_breaches"],
        "actual_breaches_per_year": var_info["actual_per_year"],
        "expected_breaches_per_year": var_info["expected_per_year"],
        "sample_years": var_info["n_years"],
    }
    (OUT_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 64)
    print(f"Hill estimator - {NAME} ({TICKER})")
    print("=" * 64)
    print(f"Sample: {summary['start']} -> {summary['end']}  ({summary['n_returns']} daily returns)")
    print(f"Positive-loss days used in Hill: {n}")
    print(f"Chosen k = {k_star}   alpha-hat = {alpha_star:.3f}")
    print(f"Tail threshold loss X_(k+1) = {summary['threshold_loss']:.4%}")
    print(f"Variance finite (alpha>2)? {variance_finite}   Kurtosis finite (alpha>4)? {kurtosis_finite}")
    print("-" * 64)
    print(f"Gaussian 99% VaR (constant mu,sigma): {var_info['var']:.4%}")
    print(f"Breaches: {var_info['n_breaches']}  (Gaussian expectation ~ {var_info['expected_breaches']:.1f})")
    print(
        f"Per year: actual {var_info['actual_per_year']:.1f} vs expected {var_info['expected_per_year']:.1f}"
    )
    print("-" * 64)
    print(f"Figures -> {FIG_DIR}")
    print(f"Summary -> {OUT_DIR / 'results_summary.json'}")


if __name__ == "__main__":
    main()
