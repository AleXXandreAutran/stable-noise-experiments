#!/usr/bin/env python3
"""Reproducible checks for second-order stable common-noise fluctuations.

The script produces five compact figures for numerical checks:
  1. the intrinsic weak-bias phase diagram for symmetric marks;
  2. a three-experiment regime atlas (weak, rank, clock);
  3. deterministic characteristic-exponent errors for a two-Pareto mixture;
  4. Monte Carlo scaling in a direct coupling benchmark;
  5. exact two-state endogenous-clock scaling under the stable-grid coupling.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gamma
from scipy.stats import levy_stable


def stable_constant(alpha: float) -> float:
    """Integral int_0^infty y^{-alpha} sin(y) dy by continuation."""
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2.0
    return float(gamma(1.0 - alpha) * np.cos(np.pi * alpha / 2.0))


def pareto_cf(u: float, beta: float) -> float:
    """CF of a symmetrized Pareto(beta) magnitude supported on [1,infinity)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        value, _ = quad(
            lambda x: beta * x ** (-beta - 1.0),
            1.0,
            np.inf,
            weight="cos",
            wvar=abs(u),
            epsabs=2e-12,
            epsrel=2e-12,
            limit=500,
        )
    return float(value)


def mixture_exponent_error(
    alpha: float, rho: float, sizes: np.ndarray, weight: float = 0.7, t: float = 1.0
) -> np.ndarray:
    """Absolute error in n log(phi_U(t n^{-1/alpha})) + A |t|^alpha."""
    leading = weight * stable_constant(alpha)
    errors = []
    for n in sizes:
        u = t * float(n) ** (-1.0 / alpha)
        phi = weight * pareto_cf(u, alpha) + (1.0 - weight) * pareto_cf(
            u, alpha + rho
        )
        if not 0.0 < phi <= 1.0:
            raise RuntimeError(f"quadrature returned an invalid CF value {phi} at n={n}")
        errors.append(abs(float(n) * np.log(phi) + leading * abs(t) ** alpha))
    return np.asarray(errors)


def predicted_weak_exponent(alpha: float, rho: float) -> float:
    return min(rho / alpha, (2.0 - alpha) / alpha, 1.0)


def make_phase_diagram(outdir: Path) -> None:
    alphas = np.linspace(0.15, 1.95, 500)
    rhos = np.linspace(0.02, 1.5, 420)
    aa, rr = np.meshgrid(alphas, rhos)
    delta = np.minimum.reduce([rr / aa, (2.0 - aa) / aa, np.ones_like(aa)])
    regime = np.where(delta > 0.5, 2.0, np.where(delta < 0.5, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.contourf(
        aa,
        rr,
        regime,
        levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["#d95f59", "#f4cf64", "#63b7a6"],
        alpha=0.92,
    )
    a_left = np.linspace(0.15, 4.0 / 3.0, 300)
    ax.plot(a_left, a_left / 2.0, color="#172a3a", lw=2.0, label=r"$\rho=\alpha/2$")
    ax.axvline(4.0 / 3.0, color="#172a3a", lw=1.7, ls="--", label=r"$\alpha=4/3$")
    ax.plot(alphas, 2.0 - alphas, color="#714955", lw=1.3, ls=":", label=r"$\rho=2-\alpha$")
    ax.text(0.35, 1.22, "sampling term slower", fontsize=10, color="#173b35")
    ax.text(1.53, 0.90, "stable-law bias slower", fontsize=10, color="#632c2a", rotation=90)
    ax.text(0.72, 0.12, "stable-law bias slower", fontsize=10, color="#632c2a")
    ax.set(xlabel=r"stable index $\alpha$", ylabel=r"second-order tail index $\rho$", xlim=(0.15, 1.95), ylim=(0.02, 1.5))
    ax.set_title("Intrinsic weak-bias comparison (symmetric marks)")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.savefig(outdir / "phase_diagram.pdf")
    fig.savefig(outdir / "phase_diagram.png", dpi=220)
    plt.close(fig)


def make_regime_atlas(outdir: Path) -> None:
    """Compare the three proved notions of second-order scale."""
    alphas = np.linspace(0.15, 1.95, 600)
    rhos = np.linspace(0.02, 1.50, 500)
    aa, rr = np.meshgrid(alphas, rhos)
    critical_tol = 0.008

    weak_delta = np.minimum.reduce(
        [rr / aa, (2.0 - aa) / aa, np.ones_like(aa)]
    )
    rank_delta = np.where(
        rr < 1.0 - aa / 2.0,
        rr / aa,
        (2.0 - aa) / (2.0 * aa),
    )
    clock_delta = np.broadcast_to(1.0 / (2.0 * aa), aa.shape)

    def classify(delta: np.ndarray) -> np.ndarray:
        return np.where(
            delta < 0.5 - critical_tol,
            0.0,
            np.where(delta > 0.5 + critical_tol, 2.0, 1.0),
        )

    fields = [classify(weak_delta), classify(rank_delta), classify(clock_delta)]
    titles = [
        "Enhanced weak-law defect",
        "Monotone-rank coupling",
        r"Stable-grid clock ($\gamma=1/2$)",
    ]
    colors = ["#d95f59", "#f4cf64", "#63b7a6"]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.5, 4.6), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.82, wspace=0.04)
    for ax, field, title in zip(axes, fields, titles):
        ax.contourf(
            aa,
            rr,
            field,
            levels=[-0.5, 0.5, 1.5, 2.5],
            colors=colors,
            alpha=0.94,
        )
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(r"stable index $\alpha$")
        ax.set_xlim(0.15, 1.95)
        ax.set_ylim(0.02, 1.50)
        ax.axvline(1.0, color="#172a3a", lw=1.0, ls="--")
    axes[0].set_ylabel(r"second-order tail index $\rho$")
    axes[0].plot(
        alphas,
        2.0 - alphas,
        color="#714955",
        lw=1.1,
        ls=":",
        label=r"$\rho=2-\alpha$",
    )
    axes[0].plot(
        alphas[alphas <= 4.0 / 3.0],
        alphas[alphas <= 4.0 / 3.0] / 2.0,
        color="#172a3a",
        lw=1.4,
    )
    axes[1].plot(
        alphas,
        np.maximum(1.0 - alphas / 2.0, 0.0),
        color="#714955",
        lw=1.1,
        ls=":",
        label=r"$\rho=1-\alpha/2$",
    )
    axes[1].plot(
        alphas[alphas <= 1.0],
        alphas[alphas <= 1.0] / 2.0,
        color="#172a3a",
        lw=1.4,
    )
    axes[2].text(
        1.02,
        1.42,
        r"raw asymmetric $\alpha=1$:" "\n" r"$\log N$ enhancement",
        fontsize=7.8,
        color="#172a3a",
        va="top",
        linespacing=1.05,
    )
    legend = [
        Patch(facecolor=colors[0], label="driver/approximation slower"),
        Patch(facecolor=colors[1], label="nominally critical"),
        Patch(facecolor=colors[2], label="sampling slower"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Three inequivalent second-order experiments", fontsize=12, y=0.96)
    fig.savefig(outdir / "regime_atlas.pdf")
    fig.savefig(outdir / "regime_atlas.png", dpi=220)
    plt.close(fig)


def make_cf_checks(outdir: Path) -> dict[str, float]:
    sizes = np.unique(np.logspace(2.0, 5.0, 22).astype(int))
    cases = [
        (1.20, 0.25, "tail term", "#d95f59"),
        (1.20, 1.00, "analytic term", "#4385be"),
        (1.60, 1.00, "analytic term, heavy alpha", "#5c9f73"),
        (4.0 / 3.0, 2.0 / 3.0, "log boundary", "#8f67a8"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    slopes: dict[str, float] = {}
    for alpha, rho, label, color in cases:
        err = mixture_exponent_error(alpha, rho, sizes)
        fit_slice = slice(-9, None)
        slope = float(np.polyfit(np.log(sizes[fit_slice]), np.log(err[fit_slice]), 1)[0])
        key = f"alpha={alpha:.6g},rho={rho:.6g}"
        slopes[key] = slope
        predicted = predicted_weak_exponent(alpha, rho)
        boundary = abs(rho - (2.0 - alpha)) < 1e-10
        reference_label = "nominal" if boundary else "theory"
        ax.loglog(
            sizes,
            err,
            "o-",
            ms=3.2,
            lw=1.25,
            color=color,
            label=rf"{label}: $(\alpha,\rho)=({alpha:.2f},{rho:.2f})$, fit {slope:.2f}, {reference_label} $-{predicted:.2f}$",
        )
    ax.set(xlabel=r"number of summands $N$", ylabel=r"$|N\log\varphi_U(N^{-1/\alpha})+A_\alpha|$")
    ax.set_title("Deterministic characteristic-exponent checks")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "cf_error.pdf")
    fig.savefig(outdir / "cf_error.png", dpi=220)
    plt.close(fig)
    return slopes


def location_robust_scale(
    sizes: np.ndarray,
    gamma_env: float,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Median absolute error for the direct coupling benchmark.

    A robust scale is essential because the alpha-stable environmental
    correction has no second moment.
    """
    out = []
    chunk = 300
    for n in sizes:
        errors = []
        done = 0
        while done < repetitions:
            m = min(chunk, repetitions - done)
            common = levy_stable.rvs(1.5, 0.0, size=m, random_state=rng)
            correction = levy_stable.rvs(1.5, 0.0, size=m, random_state=rng)
            shifted = common + float(n) ** (-gamma_env) * correction
            # For phi(x)=x and xi_i~N(0,1), the sample mean is exactly
            # N(0,1/N); generating that sufficient statistic avoids a large array.
            empirical = shifted + rng.standard_normal(m) / np.sqrt(float(n))
            target = common
            errors.append(np.abs(empirical - target))
            done += m
        out.append(float(np.median(np.concatenate(errors))))
    return np.asarray(out)


def make_location_checks(outdir: Path, quick: bool) -> dict[str, float]:
    rng = np.random.default_rng(20260811)
    sizes = np.asarray([100, 200, 400, 800, 1600, 3200] if quick else [100, 200, 400, 800, 1600, 3200, 6400])
    repetitions = 900 if quick else 2200
    cases = [(0.75, "Gaussian-dominated", "#4385be"), (0.50, "boundary", "#8f67a8"), (0.25, "stable-dominated", "#d95f59")]
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    slopes: dict[str, float] = {}
    for gamma_env, label, color in cases:
        robust_scale = location_robust_scale(sizes, gamma_env, repetitions, rng)
        slope = float(np.polyfit(np.log(sizes[-4:]), np.log(robust_scale[-4:]), 1)[0])
        slopes[f"gamma={gamma_env:.2f}"] = slope
        predicted = min(0.5, gamma_env)
        ax.loglog(
            sizes,
            robust_scale,
            "o-",
            color=color,
            lw=1.5,
            ms=4,
            label=rf"{label}: $\gamma={gamma_env:.2f}$, fit {slope:.2f}, theory $-{predicted:.2f}$",
        )
    ax.set(xlabel=r"particle number $N$", ylabel="median absolute error")
    ax.set_title(r"Direct coupling benchmark: $N^{-1/2}$ versus $N^{-\gamma}$")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(outdir / "location_scaling.pdf")
    fig.savefig(outdir / "location_scaling.png", dpi=220)
    plt.close(fig)
    return slopes


def endogenous_clock_scale(
    sizes: np.ndarray,
    alpha: float,
    p: float,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Median |S_{K/N}-S_p| for the canonical stable-grid coupling.

    Conditional on K, symmetry and stable self-similarity give the exact
    distribution sign(K/N-p) |K/N-p|^(1/alpha) xi, with xi standard S-alpha-S.
    """
    scales = []
    for n in sizes:
        k = rng.binomial(int(n), p, size=repetitions)
        xi = levy_stable.rvs(alpha, 0.0, size=repetitions, random_state=rng)
        delta = k / float(n) - p
        coupled_error = np.sign(delta) * np.abs(delta) ** (1.0 / alpha) * xi
        scales.append(float(np.median(np.abs(coupled_error))))
    return np.asarray(scales)


def make_endogenous_clock_checks(outdir: Path, quick: bool) -> dict[str, float]:
    rng = np.random.default_rng(20260812)
    sizes = np.asarray(
        [200, 400, 800, 1600, 3200, 6400]
        if quick
        else [200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    )
    repetitions = 5000 if quick else 18000
    p = 1.0 - np.exp(-1.0)
    cases = [
        (0.75, "#4385be"),
        (1.00, "#8f67a8"),
        (1.50, "#d95f59"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    slopes: dict[str, float] = {}
    for alpha, color in cases:
        scale = endogenous_clock_scale(sizes, alpha, p, repetitions, rng)
        slope = float(np.polyfit(np.log(sizes[-4:]), np.log(scale[-4:]), 1)[0])
        slopes[f"alpha={alpha:.2f}"] = slope
        predicted = -1.0 / (2.0 * alpha)
        axes[0].loglog(
            sizes,
            scale,
            "o-",
            color=color,
            lw=1.45,
            ms=3.8,
            label=rf"$\alpha={alpha:.2f}$: fit {slope:.2f}, theory {predicted:.2f}",
        )
    axes[0].set(
        xlabel=r"particle number $N$",
        ylabel=r"median $|S_{K/N}-S_p|$",
        title="Microscopic coupling displacement",
    )
    axes[0].grid(True, which="both", alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)

    # The exact marginal characteristic-function defect from (8.46) at u=1.
    weak_error = np.abs(
        sizes * np.log(1.0 - p + p * np.exp(-1.0 / sizes)) + p
    )
    weak_slope = float(
        np.polyfit(np.log(sizes[-4:]), np.log(weak_error[-4:]), 1)[0]
    )
    slopes["weak_cf"] = weak_slope
    axes[1].loglog(
        sizes,
        weak_error,
        "o-",
        color="#327a68",
        lw=1.5,
        ms=4,
        label=rf"exact weak defect: fit {weak_slope:.2f}, theory $-1$",
    )
    reference = weak_error[-1] * (sizes / sizes[-1]) ** (-1.0)
    axes[1].loglog(sizes, reference, "--", color="#172a3a", lw=1.0)
    axes[1].set(
        xlabel=r"particle number $N$",
        ylabel="absolute exponent defect",
        title="Same model: marginal weak error",
    )
    axes[1].grid(True, which="both", alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Two-state endogenous clock: coupled and weak scales differ")
    fig.savefig(outdir / "endogenous_clock_scaling.pdf")
    fig.savefig(outdir / "endogenous_clock_scaling.png", dpi=220)
    plt.close(fig)
    return slopes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--quick", action="store_true", help="use the CI-sized Monte Carlo run")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    make_phase_diagram(args.outdir)
    make_regime_atlas(args.outdir)
    check_report = {
        "characteristic_exponent_fitted_slopes": make_cf_checks(args.outdir),
        "direct_coupling_fitted_slopes": make_location_checks(args.outdir, args.quick),
        "endogenous_clock_fitted_slopes": make_endogenous_clock_checks(
            args.outdir, args.quick
        ),
        "seeds": [20260811, 20260812],
        "note": "Fitted slopes are numerical checks only; theorem predictions are analytic.",
    }
    (args.outdir / "checks.json").write_text(json.dumps(check_report, indent=2) + "\n")
    print(json.dumps(check_report, indent=2))


if __name__ == "__main__":
    main()
