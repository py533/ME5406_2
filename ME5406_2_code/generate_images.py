"""
generate_images.py — SCI-quality training curves and evaluation figures.

Output (all saved to ./images/):
    figure_a.png   Episode reward training curve (SAC vs TD3)
    figure_b.png   Grasp & reach success rate vs training steps (SAC vs TD3)
    figure_c.png   Demo evaluation bar chart — grasp rate (100 episodes each)
   
Run:
    python3 generate_images.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("images", exist_ok=True)

# ── file paths ────────────────────────────────────────────────────────────────
SAC_LOG         = "./logs/monitor_sac.csv"
TD3_LOG         = "./logs/monitor_td3.csv"
SAC_SUCCESS_LOG = "./logs/success_sac.csv"
TD3_SUCCESS_LOG = "./logs/success_td3.csv"
DEMO_EVAL_CSV   = "./logs/demo_eval.csv"

# ── SCI colour palette (Paul Tol colorblind-safe) ────────────────────────────
SAC_COLOR  = "#CC3311"   # vermillion
TD3_COLOR  = "#0077BB"   # teal-blue
SAC_ALPHA  = 0.18
TD3_ALPHA  = 0.18

# ── global rcParams — SCI journal style ──────────────────────────────────────
plt.rcParams.update({
    # typography
    "font.family":            "serif",
    "font.serif":             ["Times New Roman", "DejaVu Serif"],
    "font.size":              9,
    "axes.titlesize":         9.5,
    "axes.labelsize":         9,
    "xtick.labelsize":        8,
    "ytick.labelsize":        8,
    "legend.fontsize":        8,
    "legend.title_fontsize":  8.5,
    "mathtext.fontset":       "dejavuserif",
    # lines
    "lines.linewidth":        1.6,
    "lines.markersize":       4,
    "patch.linewidth":        0.7,
    "axes.linewidth":         0.8,
    # background — clean white, no grid
    "axes.facecolor":         "white",
    "figure.facecolor":       "white",
    "axes.grid":              False,
    # ticks — inward, both axes
    "xtick.direction":        "in",
    "ytick.direction":        "in",
    "xtick.major.width":      0.8,
    "ytick.major.width":      0.8,
    "xtick.major.size":       4.0,
    "ytick.major.size":       4.0,
    "xtick.minor.visible":    True,
    "ytick.minor.visible":    True,
    "xtick.minor.width":      0.5,
    "ytick.minor.width":      0.5,
    "xtick.minor.size":       2.0,
    "ytick.minor.size":       2.0,
    # output
    "figure.dpi":             300,
    "savefig.dpi":            300,
    "savefig.bbox":           "tight",
    "savefig.pad_inches":     0.04,
    # legend — minimal frame
    "legend.framealpha":      0.9,
    "legend.edgecolor":       "0.82",
    "legend.borderpad":       0.45,
    "legend.labelspacing":    0.35,
    "legend.handlelength":    1.8,
    "legend.handletextpad":   0.5,
    "legend.columnspacing":   1.0,
    "legend.fancybox":        False,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_monitor(base: str) -> str:
    m = base + ".monitor.csv"
    return m if os.path.exists(m) else base


def _load_monitor(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=1)
    df["episode"]  = np.arange(1, len(df) + 1)
    df["timestep"] = df["l"].cumsum()
    return df


def _smooth(values: np.ndarray, sigma: float = 3.5) -> np.ndarray:
    return gaussian_filter1d(values.astype(float), sigma=sigma)


def _rolling_band(values: np.ndarray, w: int = 30):
    n  = len(values)
    lo = np.empty(n); hi = np.empty(n)
    for i in range(n):
        win  = values[max(0, i - w + 1): i + 1].astype(float)
        mu   = win.mean(); s = win.std(ddof=0)
        lo[i] = mu - s; hi[i] = mu + s
    return lo, hi


def _kstep_fmt(ax) -> None:
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))


def _decorate(ax, xlabel: str, ylabel: str, ylim=None) -> None:
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    # only left and bottom spines (SCI standard)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if ylim is not None:
        ax.set_ylim(ylim)
    # subtle horizontal reference lines only
    ax.yaxis.set_tick_params(which="both", right=False)
    ax.xaxis.set_tick_params(which="both", top=False)


# ── Figure a: Episode Reward ──────────────────────────────────────────────────

def _plot_figure_a() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    plotted = False
    for path, label, color in [
        (_resolve_monitor(SAC_LOG), "SAC", SAC_COLOR),
        (_resolve_monitor(TD3_LOG), "TD3", TD3_COLOR),
    ]:
        if not os.path.exists(path):
            continue
        try:
            df  = _load_monitor(path)
            raw = df["r"].values
            ts  = df["timestep"].values
            sm  = _smooth(raw, sigma=4.0)

            ax.plot(ts, sm, color=color, linewidth=1.6,
                    label=label, zorder=3)
            plotted = True
        except Exception as e:
            print(f"  Warning [{label} reward]: {e}")

    _kstep_fmt(ax)
    _decorate(ax, xlabel="Training Steps", ylabel="Episode Reward")
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle=":", zorder=1)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    if plotted:
        ax.legend(loc="lower right", ncol=1)
    else:
        ax.text(0.5, 0.5, "No monitor logs found.\nRun main.py first.",
                ha="center", va="center", fontsize=7, color="gray",
                transform=ax.transAxes)

    plt.savefig("images/figure_a.png")
    plt.close()
    print("  Saved → images/figure_a.png")


# ── Figure c: Episode Length over Training ────────────────────────────────────

def _plot_figure_c() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    plotted = False
    for path, label, color in [
        (_resolve_monitor(SAC_LOG), "SAC", SAC_COLOR),
        (_resolve_monitor(TD3_LOG), "TD3", TD3_COLOR),
    ]:
        if not os.path.exists(path):
            continue
        try:
            df  = _load_monitor(path)
            raw = df["l"].values.astype(float)
            ts  = df["timestep"].values
            sm  = _smooth(raw, sigma=4.0)

            ax.plot(ts, sm, color=color, linewidth=1.6,
                    label=label, zorder=3)
            plotted = True
        except Exception as e:
            print(f"  Warning [{label} steps]: {e}")

    _kstep_fmt(ax)
    _decorate(ax, xlabel="Training Steps", ylabel="Steps per Episode", ylim=(0, None))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    if plotted:
        ax.legend(loc="upper right", ncol=1)
    else:
        ax.text(0.5, 0.5, "No monitor logs found.\nRun main.py first.",
                ha="center", va="center", fontsize=7, color="gray",
                transform=ax.transAxes)

    plt.savefig("images/figure_c.png")
    plt.close()
    print("  Saved → images/figure_c.png")


# ── Figure b: Success Rate vs Training Steps ──────────────────────────────────

def _plot_figure_b() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    plotted = False
    legend_handles: list = []

    for path, label, color in [
        (SAC_SUCCESS_LOG, "SAC", SAC_COLOR),
        (TD3_SUCCESS_LOG, "TD3", TD3_COLOR),
    ]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            if len(df) < 3:
                continue
            x         = df["timesteps"].values
            grasp_raw = df["grasp_rate"].values
            reach_raw = df["reach_rate"].values
            grasp_sm  = _smooth(grasp_raw, sigma=2.5)
            reach_sm  = _smooth(reach_raw, sigma=2.5)

            ax.plot(x, grasp_sm, color=color, linewidth=1.6, zorder=3)
            ax.plot(x, reach_sm, color=color, linewidth=1.0,
                    linestyle="--", alpha=0.75, zorder=3)

            legend_handles += [
                Line2D([0], [0], color=color, linewidth=1.6,
                       label=f"{label} grasp"),
                Line2D([0], [0], color=color, linewidth=1.0,
                       linestyle="--", alpha=0.75,
                       label=f"{label} reach"),
            ]
            plotted = True
        except Exception as e:
            print(f"  Warning [{label} success]: {e}")

    _kstep_fmt(ax)
    _decorate(ax,
              xlabel="Training Steps",
              ylabel="Success Rate (%)",
              ylim=(-3, 108))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))

    if plotted and legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", ncol=2)
    else:
        ax.text(0.5, 0.5, "No success logs found.\nRun main.py first.",
                ha="center", va="center", fontsize=7, color="gray",
                transform=ax.transAxes)

    plt.savefig("images/figure_b.png")
    plt.close()
    print("  Saved → images/figure_b.png")




# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Generating SCI-quality figures …\n")
    _plot_figure_a()
    _plot_figure_c()
    _plot_figure_b()
    print("\n  All figures saved to ./images/\n")
