#!/usr/bin/env python3
"""Second-order checks for an emergent alpha-stable common noise.

The experiments distinguish weak-law, rank-coupling, location,
and clock errors. 
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CACHE = Path(tempfile.gettempdir()) / "autran_stable_noise_mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gamma as gamma_function
from scipy.stats import levy_stable


COLORS = {
    "red": "#C84B4B",
    "blue": "#3274A1",
    "green": "#3A8D71",
    "purple": "#8064A2",
    "gold": "#D6A43A",
    "ink": "#243447",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def stable_constant(alpha: float) -> float:
    """Return integral_0^infinity y^(-alpha) sin(y) dy by continuation."""
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2.0
    return float(gamma_function(1.0 - alpha) * np.cos(np.pi * alpha / 2.0))


@lru_cache(maxsize=None)
def _pareto_defect_tail(beta: float) -> float:
    """Integral from one to infinity in the scaled 1-cos representation."""
    integrand = lambda y: beta * y ** (-beta - 1.0) * 2.0 * np.sin(y / 2.0) ** 2
    if beta < 2.0:
        # The full integral is the stable constant. Subtracting the smooth
        # interval [0,1] avoids an oscillatory infinite-range quadrature.
        small, _ = quad(integrand, 0.0, 1.0, epsabs=2e-13, epsrel=2e-13)
        return stable_constant(beta) - float(small)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        tail, _ = quad(
            integrand, 1.0, np.inf, epsabs=2e-12, epsrel=2e-12, limit=2000
        )
    return float(tail)


def symmetrized_pareto_cf_defect(u: float, beta: float) -> float:
    """Return 1-CF(eps*R) without subtracting two nearly equal numbers."""
    u_abs = abs(float(u))
    if u_abs == 0.0:
        return 0.0
    integrand = lambda y: beta * y ** (-beta - 1.0) * 2.0 * np.sin(y / 2.0) ** 2
    if u_abs < 1.0:
        local, _ = quad(
            integrand, u_abs, 1.0, epsabs=2e-13, epsrel=2e-13, limit=600
        )
        scaled_integral = float(local) + _pareto_defect_tail(beta)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            scaled_integral, _ = quad(
                integrand,
                u_abs,
                np.inf,
                epsabs=2e-12,
                epsrel=2e-12,
                limit=2000,
            )
    return u_abs ** beta * float(scaled_integral)


def weak_exponent_error(
    alpha: float,
    rho: float,
    sizes: np.ndarray,
    weight: float = 0.7,
) -> np.ndarray:
    """Compute the signed-exponent error for a two-Pareto mixture."""
    leading = weight * stable_constant(alpha)
    errors: list[float] = []
    for n in sizes:
        u = float(n) ** (-1.0 / alpha)
        defect = weight * symmetrized_pareto_cf_defect(u, alpha)
        defect += (1.0 - weight) * symmetrized_pareto_cf_defect(u, alpha + rho)
        if not 0.0 <= defect < 1.0:
            raise RuntimeError(f"invalid characteristic-function defect {defect} at N={n}")
        errors.append(abs(float(n) * np.log1p(-defect) + leading))
    return np.asarray(errors)


def weak_theory_power(alpha: float, rho: float) -> float:
    return min(rho, 2.0 - alpha) / alpha


def weak_checks(outdir: Path, quick: bool) -> tuple[dict, list[dict]]:
    sizes = np.unique(np.logspace(2.0, 4.7 if quick else 5.2, 18 if quick else 25).astype(int))
    cases = [
        (1.20, 0.25, "tail correction", COLORS["red"]),
        (1.20, 1.00, "analytic correction", COLORS["blue"]),
        (1.60, 1.00, "heavy alpha", COLORS["green"]),
        (4.0 / 3.0, 2.0 / 3.0, "logarithmic boundary", COLORS["purple"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    results: dict[str, dict[str, float | bool]] = {}
    checks: list[dict] = []
    for alpha, rho, label, color in cases:
        errors = weak_exponent_error(alpha, rho, sizes)
        fit = slice(-8, None)
        fitted = float(np.polyfit(np.log(sizes[fit]), np.log(errors[fit]), 1)[0])
        power = weak_theory_power(alpha, rho)
        boundary = abs(rho - (2.0 - alpha)) < 1e-10
        reference = sizes.astype(float) ** (-power)
        if boundary:
            reference *= np.log(sizes)
        expected_fit = float(
            np.polyfit(np.log(sizes[fit]), np.log(reference[fit]), 1)[0]
        )
        tolerance = 0.075 if not quick else 0.11
        passed = abs(fitted - expected_fit) <= tolerance
        key = f"alpha={alpha:.6g},rho={rho:.6g}"
        results[key] = {
            "fitted_loglog_slope": fitted,
            "asymptotic_power_without_sign": power,
            "finite_grid_expected_slope": expected_fit,
            "log_boundary": boundary,
            "tolerance": tolerance,
            "passed": passed,
        }
        checks.append({"name": f"weak slope {key}", "passed": passed})
        ax.loglog(
            sizes,
            errors,
            "o-",
            ms=3.3,
            lw=1.35,
            color=color,
            label=f"{label}: fit {fitted:.3f}, reference {expected_fit:.3f}",
        )
    ax.set_xlabel("number of summands N")
    ax.set_ylabel(r"$|N\log\varphi(N^{-1/\alpha})+A_\alpha|$")
    ax.set_title("Weak characteristic-exponent rates")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "weak_characteristic_rates")
    return results, checks


def mixture_quantile(
    v: np.ndarray, alpha: float, rho: float, weight: float
) -> np.ndarray:
    """Invert weight*x^-alpha+(1-weight)*x^(-alpha-rho)=v.

    The quantile is set to zero for v>=1, as in the Poisson-rank construction.
    Newton iteration is vectorized and starts from the second-order inverse-tail
    expansion.
    """
    values = np.asarray(v, dtype=float)
    out = np.zeros_like(values)
    mask = (values > 0.0) & (values < 1.0)
    if not np.any(mask):
        return out
    vm = np.maximum(values[mask], np.finfo(float).tiny)
    d = (1.0 - weight) / weight
    leading = (weight / vm) ** (1.0 / alpha)
    q = leading * (1.0 + (d / alpha) * leading ** (-rho))
    q = np.maximum(q, 1.0)
    for _ in range(12):
        tail = weight * q ** (-alpha) + (1.0 - weight) * q ** (-alpha - rho)
        derivative = -alpha * weight * q ** (-alpha - 1.0)
        derivative -= (alpha + rho) * (1.0 - weight) * q ** (-alpha - rho - 1.0)
        candidate = q - (tail - vm) / derivative
        q = np.where(candidate >= 1.0, candidate, 0.5 * (q + 1.0))
    out[mask] = q
    return out


def rank_integrand_difference(
    n: int, y: np.ndarray, alpha: float, rho: float, weight: float
) -> np.ndarray:
    y_safe = np.maximum(np.asarray(y, dtype=float), np.finfo(float).tiny)
    g = (weight / y_safe) ** (1.0 / alpha)
    g_n = float(n) ** (-1.0 / alpha) * mixture_quantile(
        y_safe / float(n), alpha, rho, weight
    )
    return g_n - g


def rank_tail_variance(
    n: int,
    y_cut: float,
    alpha: float,
    rho: float,
    weight: float,
) -> float:
    """Bracket of ranks above y_cut; used for the Gaussian tail completion."""
    if y_cut >= n:
        return float(
            weight ** (2.0 / alpha)
            * y_cut ** (1.0 - 2.0 / alpha)
            / (2.0 / alpha - 1.0)
        )
    stop = float(n) * (1.0 - 1e-11)
    grid = np.geomspace(y_cut, stop, 2400)
    difference = rank_integrand_difference(n, grid, alpha, rho, weight)
    numerical = float(np.trapezoid(difference * difference, grid))
    beyond_n = (
        weight ** (2.0 / alpha)
        * float(n) ** (1.0 - 2.0 / alpha)
        / (2.0 / alpha - 1.0)
    )
    return max(numerical + beyond_n, 0.0)


def sample_rank_error(
    n: int,
    alpha: float,
    rho: float,
    weight: float,
    repetitions: int,
    y_cut: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample retained Poisson ranks and complete the square-integrable tail."""
    counts = rng.poisson(y_cut, size=repetitions)
    labels = np.repeat(np.arange(repetitions), counts)
    y = rng.uniform(0.0, y_cut, size=int(np.sum(counts)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=y.size)
    increments = signs * rank_integrand_difference(n, y, alpha, rho, weight)
    retained = np.bincount(labels, weights=increments, minlength=repetitions)
    tail_variance = rank_tail_variance(n, y_cut, alpha, rho, weight)
    return retained + np.sqrt(tail_variance) * rng.standard_normal(repetitions)


def rank_checks(
    outdir: Path, quick: bool, rng: np.random.Generator
) -> tuple[dict, list[dict]]:
    alpha = 1.2
    rho_c = 1.0 - alpha / 2.0
    weight = 0.7
    sizes = np.asarray(
        [400, 800, 1600, 3200, 6400, 12800]
        if quick
        else [400, 800, 1600, 3200, 6400, 12800, 25600]
    )
    repetitions = 1400 if quick else 3500
    y_cut = 70.0 if quick else 100.0
    cases = [
        (0.20, "tail-rank tangent", COLORS["red"]),
        (rho_c, "Levy--Brownian boundary", COLORS["purple"]),
        (0.80, "bulk Brownian tangent", COLORS["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    results: dict[str, dict] = {}
    checks: list[dict] = []
    for rho, label, color in cases:
        medians = []
        for n in sizes:
            errors = sample_rank_error(
                int(n), alpha, rho, weight, repetitions, y_cut, rng
            )
            medians.append(float(np.median(np.abs(errors))))
        medians_array = np.asarray(medians)
        fit_slice = slice(-5, None)
        fitted = float(
            np.polyfit(np.log(sizes[fit_slice]), np.log(medians_array[fit_slice]), 1)[0]
        )
        if rho < rho_c - 1e-12:
            normalization = sizes.astype(float) ** (-rho / alpha)
            regime = "tail"
        elif abs(rho - rho_c) <= 1e-12:
            normalization = sizes.astype(float) ** (-(2.0 - alpha) / (2.0 * alpha))
            normalization *= np.sqrt(np.log(sizes))
            regime = "boundary"
        else:
            normalization = sizes.astype(float) ** (-(2.0 - alpha) / (2.0 * alpha))
            regime = "body"
        expected_fit = float(
            np.polyfit(
                np.log(sizes[fit_slice]), np.log(normalization[fit_slice]), 1
            )[0]
        )
        tolerance = 0.10 if not quick else 0.15
        passed = abs(fitted - expected_fit) <= tolerance
        results[f"rho={rho:.3f}"] = {
            "alpha": alpha,
            "regime": regime,
            "fitted_loglog_slope": fitted,
            "finite_grid_expected_slope": expected_fit,
            "normalization_ratio_range_last_five": [
                float(np.min(medians_array[fit_slice] / normalization[fit_slice])),
                float(np.max(medians_array[fit_slice] / normalization[fit_slice])),
            ],
            "retained_rank_cutoff": y_cut,
            "repetitions": repetitions,
            "tolerance": tolerance,
            "passed": passed,
        }
        checks.append({"name": f"rank slope rho={rho:.3f}", "passed": passed})
        ax.loglog(
            sizes,
            medians_array,
            "o-",
            lw=1.4,
            ms=3.8,
            color=color,
            label=f"{label}: fit {fitted:.3f}, reference {expected_fit:.3f}",
        )
    ax.set_xlabel("Poissonized sample size N")
    ax.set_ylabel("median absolute coupled displacement")
    ax.set_title(r"Monotone rank coupling ($\alpha=1.2$, boundary $\rho_c=0.4$)")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "rank_coupling_rates")
    return results, checks


def location_scale(
    sizes: np.ndarray,
    gamma_env: float,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Median error in Xbar=L+N^-gamma Z+N^-1/2 Normal."""
    scales = []
    for n in sizes:
        environment_error = levy_stable.rvs(
            1.5, 0.0, size=repetitions, random_state=rng
        )
        sampling_error = rng.standard_normal(repetitions) / np.sqrt(float(n))
        error = float(n) ** (-gamma_env) * environment_error + sampling_error
        scales.append(float(np.median(np.abs(error))))
    return np.asarray(scales)


def location_checks(
    outdir: Path, quick: bool, rng: np.random.Generator
) -> tuple[dict, list[dict]]:
    sizes = np.asarray(
        [100, 200, 400, 800, 1600, 3200]
        if quick
        else [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    )
    repetitions = 1800 if quick else 6000
    cases = [
        (0.75, "sampling dominated", COLORS["blue"]),
        (0.50, "critical", COLORS["purple"]),
        (0.25, "environment dominated", COLORS["red"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    results: dict[str, dict] = {}
    checks: list[dict] = []
    for gamma_env, label, color in cases:
        scales = location_scale(sizes, gamma_env, repetitions, rng)
        fitted = float(np.polyfit(np.log(sizes[-5:]), np.log(scales[-5:]), 1)[0])
        expected = -min(0.5, gamma_env)
        tolerance = 0.075 if not quick else 0.11
        passed = abs(fitted - expected) <= tolerance
        results[f"gamma={gamma_env:.2f}"] = {
            "fitted_loglog_slope": fitted,
            "expected_slope": expected,
            "repetitions": repetitions,
            "tolerance": tolerance,
            "passed": passed,
        }
        checks.append({"name": f"location slope gamma={gamma_env:.2f}", "passed": passed})
        ax.loglog(
            sizes,
            scales,
            "o-",
            color=color,
            lw=1.4,
            ms=3.8,
            label=f"{label}: fit {fitted:.3f}, theory {expected:.3f}",
        )
    ax.set_xlabel("particle number N")
    ax.set_ylabel("median absolute location error")
    ax.set_title("Location transfer: sampling versus environment")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "location_transfer_rates")
    return results, checks


def clock_scale(
    sizes: np.ndarray,
    alpha: float,
    p: float,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact stable-grid displacement scale conditional on a binomial clock."""
    medians = []
    for n in sizes:
        k = rng.binomial(int(n), p, size=repetitions)
        xi = levy_stable.rvs(alpha, 0.0, size=repetitions, random_state=rng)
        clock_error = k / float(n) - p
        displacement = np.sign(clock_error) * np.abs(clock_error) ** (1.0 / alpha) * xi
        medians.append(float(np.median(np.abs(displacement))))
    return np.asarray(medians)


def clock_checks(
    outdir: Path, quick: bool, rng: np.random.Generator
) -> tuple[dict, list[dict]]:
    sizes = np.asarray(
        [200, 400, 800, 1600, 3200, 6400]
        if quick
        else [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    )
    repetitions = 6000 if quick else 22000
    p = 1.0 - np.exp(-1.0)
    cases = [
        (0.75, COLORS["blue"], "root-N term vanishes"),
        (1.00, COLORS["purple"], "root-N critical"),
        (1.50, COLORS["red"], "root-N term grows"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), constrained_layout=True)
    results: dict[str, dict] = {}
    checks: list[dict] = []
    for alpha, color, label in cases:
        medians = clock_scale(sizes, alpha, p, repetitions, rng)
        raw_fitted = float(np.polyfit(np.log(sizes[-5:]), np.log(medians[-5:]), 1)[0])
        normalized = np.sqrt(sizes) * medians
        normalized_fitted = float(
            np.polyfit(np.log(sizes[-5:]), np.log(normalized[-5:]), 1)[0]
        )
        raw_expected = -1.0 / (2.0 * alpha)
        normalized_expected = 0.5 - 1.0 / (2.0 * alpha)
        tolerance = 0.065 if not quick else 0.095
        passed = (
            abs(raw_fitted - raw_expected) <= tolerance
            and abs(normalized_fitted - normalized_expected) <= tolerance
        )
        results[f"alpha={alpha:.2f}"] = {
            "raw_fitted_slope": raw_fitted,
            "raw_expected_slope": raw_expected,
            "rootN_fitted_slope": normalized_fitted,
            "rootN_expected_slope": normalized_expected,
            "regime": label,
            "repetitions": repetitions,
            "tolerance": tolerance,
            "passed": passed,
        }
        checks.append({"name": f"clock slopes alpha={alpha:.2f}", "passed": passed})
        axes[0].loglog(
            sizes,
            medians,
            "o-",
            color=color,
            lw=1.4,
            ms=3.8,
            label=f"alpha={alpha:.2f}: fit {raw_fitted:.3f}",
        )
        axes[1].loglog(
            sizes,
            normalized,
            "o-",
            color=color,
            lw=1.4,
            ms=3.8,
            label=f"alpha={alpha:.2f}: fit {normalized_fitted:+.3f}",
        )
    axes[0].set_xlabel("particle number N")
    axes[0].set_ylabel(r"median $|S_{K/N}-S_p|$")
    axes[0].set_title("Raw stable-grid displacement")
    axes[1].set_xlabel("particle number N")
    axes[1].set_ylabel(r"$\sqrt{N}$ times median displacement")
    axes[1].set_title("Sampling-scale regimes")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.18)
        ax.legend(frameon=False)
    save_figure(fig, outdir, "endogenous_clock_regimes")
    return results, checks


def regime_atlas(outdir: Path) -> None:
    alphas = np.linspace(0.18, 1.94, 500)
    rhos = np.linspace(0.02, 1.48, 420)
    aa, rr = np.meshgrid(alphas, rhos)
    weak_power = np.minimum(rr / aa, (2.0 - aa) / aa)
    rank_power = np.where(
        rr < 1.0 - aa / 2.0,
        rr / aa,
        (2.0 - aa) / (2.0 * aa),
    )
    clock_power = np.broadcast_to(1.0 / (2.0 * aa), aa.shape)

    def classify(power: np.ndarray) -> np.ndarray:
        return np.where(power < 0.495, 0.0, np.where(power > 0.505, 2.0, 1.0))

    fields = [classify(weak_power), classify(rank_power), classify(clock_power)]
    titles = ["weak exponent", "monotone-rank coupling", "endogenous clock"]
    colors = ["#D86A63", "#F1CC73", "#69B6A4"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.84, wspace=0.05)
    for ax, field, title in zip(axes, fields, titles):
        ax.contourf(aa, rr, field, levels=[-0.5, 0.5, 1.5, 2.5], colors=colors)
        ax.axvline(1.0, color=COLORS["ink"], ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"stable index $\alpha$")
        ax.set_xlim(alphas.min(), alphas.max())
        ax.set_ylim(rhos.min(), rhos.max())
    axes[0].set_ylabel(r"second-order index $\rho$")
    axes[0].plot(alphas, 2.0 - alphas, ":", color=COLORS["ink"], lw=1.2)
    axes[0].plot(alphas[alphas <= 4 / 3], alphas[alphas <= 4 / 3] / 2, color=COLORS["ink"], lw=1.3)
    axes[1].plot(alphas, np.maximum(1.0 - alphas / 2.0, 0.0), ":", color=COLORS["ink"], lw=1.2)
    axes[1].plot(alphas[alphas <= 1], alphas[alphas <= 1] / 2, color=COLORS["ink"], lw=1.3)
    handles = [
        Patch(facecolor=colors[0], label="noise/coupling term slower"),
        Patch(facecolor=colors[1], label="critical"),
        Patch(facecolor=colors[2], label="sampling term slower"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Different second-order experiments have different boundaries", y=0.97)
    save_figure(fig, outdir, "regime_atlas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--quick", action="store_true", help="use a CI-sized run")
    parser.add_argument("--strict", action="store_true", help="fail if a check misses its recorded tolerance")
    parser.add_argument(
        "--only",
        choices=["all", "weak", "rank", "location", "clock", "atlas"],
        default="all",
        help="run one experiment or the complete suite",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(3)
    rng_rank, rng_location, rng_clock = [np.random.default_rng(s) for s in child_seeds]
    check_report: dict[str, object] = {
        "base_seed": args.seed,
        "quick": args.quick,
        "experiments": {},
        "validation": {},
        "caveat": "Numerical slopes provide checks, but do not prove, the analytic limits.",
    }
    checks: list[dict] = []

    if args.only in ("all", "weak"):
        result, local_checks = weak_checks(args.outdir, args.quick)
        check_report["experiments"]["weak_characteristic_exponent"] = result
        checks.extend(local_checks)
    if args.only in ("all", "rank"):
        result, local_checks = rank_checks(args.outdir, args.quick, rng_rank)
        check_report["experiments"]["monotone_rank_coupling"] = result
        checks.extend(local_checks)
    if args.only in ("all", "location"):
        result, local_checks = location_checks(args.outdir, args.quick, rng_location)
        check_report["experiments"]["location_transfer"] = result
        checks.extend(local_checks)
    if args.only in ("all", "clock"):
        result, local_checks = clock_checks(args.outdir, args.quick, rng_clock)
        check_report["experiments"]["endogenous_clock"] = result
        checks.extend(local_checks)
    if args.only in ("all", "atlas"):
        regime_atlas(args.outdir)

    passed = all(bool(check["passed"]) for check in checks)
    check_report["validation"] = {"passed": passed, "checks": checks}
    checks_path = args.outdir / "checks.json"
    checks_path.write_text(json.dumps(check_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check_report, indent=2))
    if args.strict and not passed:
        raise SystemExit("one or more numerical checks missed tolerance")


if __name__ == "__main__":
    main()
