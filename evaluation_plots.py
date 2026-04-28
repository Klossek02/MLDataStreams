"""Generate evaluation plots from the streaming experiment CSV files.

The script reads the detailed result files from ``results/`` and writes a small
set of presentation-ready PNG charts. It is intentionally opinionated: instead
of plotting every experiment, it highlights the baselines and drift-aware
variants that best explain the evaluation.

Usage:
    python evaluation_plots.py
    python evaluation_plots.py --results-dir results --out-dir latex/presentation/results/evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


REAL_DATASETS = ["avazu", "criteo"]
SYNTHETIC_DATASETS = ["agrawal_sudden", "agrawal_gradual"]

REAL_METHODS = [
    ("HT", "none"),
    ("SRP", "static_topk"),
    ("SRP", "online_ranking"),
    ("DASRP", "drift_aware_srp"),
]

SYNTHETIC_METHODS = [
    ("HAT", "none"),
    ("HT", "online_ranking"),
    ("SRP", "none"),
    ("DASRP", "drift_aware_srp"),
]

METHOD_ORDER = [
    "HT none",
    "HAT none",
    "SRP static top-k",
    "SRP online ranking",
    "SRP none",
    "DASRP drift-aware SRP",
    "HT drift-aware selector",
    "HAT drift-aware selector",
]

PALETTE = {
    "HT none": "#6b7280",
    "HT online ranking": "#0891b2",
    "HAT none": "#4b5563",
    "SRP static top-k": "#2563eb",
    "SRP online ranking": "#0f766e",
    "SRP none": "#7c3aed",
    "DASRP drift-aware SRP": "#dc2626",
    "HT drift-aware selector": "#ea580c",
    "HAT drift-aware selector": "#d97706",
}


def selector_label(selector: str) -> str:
    return {
        "none": "none",
        "static_topk": "static top-k",
        "online_ranking": "online ranking",
        "drift_aware_selector": "drift-aware selector",
        "drift_aware_srp": "drift-aware SRP",
    }.get(selector, selector.replace("_", " "))


def method_label(row: pd.Series) -> str:
    return f"{row['model']} {selector_label(row['selector'])}"


def load_csv(path: Path, suite: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["suite"] = suite
    return df


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_parts = []
    long_parts = []
    drift_parts = []

    files = [
        ("baseline", "results_summary.csv", "results_long.csv", "results_drifts.csv"),
        ("drift-aware", "driftaware_summary.csv", "driftaware_long.csv", "driftaware_drifts.csv"),
    ]

    for suite, summary_name, long_name, drift_name in files:
        summary_path = results_dir / summary_name
        long_path = results_dir / long_name
        drift_path = results_dir / drift_name

        if summary_path.exists():
            summary_parts.append(load_csv(summary_path, suite))
        if long_path.exists():
            long_parts.append(load_csv(long_path, suite))
        if drift_path.exists():
            drift_parts.append(load_csv(drift_path, suite))

    if not summary_parts or not long_parts:
        raise FileNotFoundError(
            f"Expected result CSVs in {results_dir.resolve()}."
        )

    summary = pd.concat(summary_parts, ignore_index=True)
    long = pd.concat(long_parts, ignore_index=True)
    drifts = pd.concat(drift_parts, ignore_index=True) if drift_parts else pd.DataFrame()

    numeric_summary = [
        "instances",
        "elapsed_ms",
        "drifts",
        "final_Accuracy",
        "final_LogLoss",
        "final_AUC",
        "final_Windowed[1000]-Accuracy",
        "final_Windowed[1000]-LogLoss",
    ]
    numeric_long = [
        "instance",
        "Accuracy",
        "LogLoss",
        "AUC",
        "Windowed[1000]-Accuracy",
        "Windowed[1000]-LogLoss",
    ]
    for col in numeric_summary:
        if col in summary:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    for col in numeric_long:
        if col in long:
            long[col] = pd.to_numeric(long[col], errors="coerce")
    if not drifts.empty and "instance" in drifts:
        drifts["instance"] = pd.to_numeric(drifts["instance"], errors="coerce")

    summary["method"] = summary.apply(method_label, axis=1)
    long["method"] = long.apply(method_label, axis=1)
    if not drifts.empty:
        drifts["method"] = drifts.apply(method_label, axis=1)

    summary["elapsed_s"] = summary["elapsed_ms"] / 1000.0
    return summary, long, drifts


def keep_methods(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    allowed = set(pairs)
    mask = df.apply(lambda r: (r["model"], r["selector"]) in allowed, axis=1)
    return df.loc[mask].copy()


def ordered_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"])
    ordered = [m for m in METHOD_ORDER if m in present]
    remaining = sorted(present - set(ordered))
    return ordered + remaining


def savefig(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def method_handles(methods: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=PALETTE.get(method, "#111827"), lw=3, label=method)
        for method in methods
    ]


def annotate_bars(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append("" if pd.isna(value) else fmt.format(value))
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)


def plot_real_final_logloss(summary: pd.DataFrame, out_dir: Path) -> None:
    data = keep_methods(summary[summary["dataset"].isin(REAL_DATASETS)], REAL_METHODS)
    order = REAL_DATASETS
    hue_order = ordered_methods(data)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(
        data=data,
        x="dataset",
        y="final_LogLoss",
        hue="method",
        order=order,
        hue_order=hue_order,
        palette=PALETTE,
        ax=ax,
    )
    annotate_bars(ax)
    ax.set_title("Final Log Loss on real CTR streams", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Final Log Loss (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="", ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.22)
    savefig(fig, out_dir, "eval_real_final_logloss.png")


def plot_real_logloss_trajectories(long: pd.DataFrame, out_dir: Path) -> None:
    data = keep_methods(long[long["dataset"].isin(REAL_DATASETS)], REAL_METHODS)
    data = data.sort_values(["dataset", "method", "instance"])
    data["smooth_logloss"] = data.groupby(["dataset", "method"])["Windowed[1000]-LogLoss"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    hue_order = ordered_methods(data)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=False)
    for ax, dataset in zip(axes, REAL_DATASETS):
        subset = data[data["dataset"] == dataset]
        sns.lineplot(
            data=subset,
            x="instance",
            y="smooth_logloss",
            hue="method",
            hue_order=hue_order,
            palette=PALETTE,
            linewidth=2.0,
            ax=ax,
            legend=False,
        )
        ax.set_title(dataset.capitalize(), fontsize=13, weight="bold")
        ax.set_xlabel("Instance")
        ax.set_ylabel("Windowed 1k Log Loss")
        ax.grid(alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x")
        sns.despine(ax=ax)

    fig.suptitle("Probability loss over time on real CTR streams", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.legend(
        handles=method_handles(hue_order),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    savefig(fig, out_dir, "eval_real_windowed_logloss.png")


def plot_synthetic_adaptation(long: pd.DataFrame, out_dir: Path) -> None:
    data = keep_methods(long[long["dataset"].isin(SYNTHETIC_DATASETS)], SYNTHETIC_METHODS)
    data = data[(data["instance"] >= 60_000) & (data["instance"] <= 140_000)].copy()
    data = data.sort_values(["dataset", "method", "instance"])
    data["smooth_logloss"] = data.groupby(["dataset", "method"])["Windowed[1000]-LogLoss"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    hue_order = ordered_methods(data)

    titles = {
        "agrawal_sudden": "Sudden drift",
        "agrawal_gradual": "Gradual drift",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, dataset in zip(axes, SYNTHETIC_DATASETS):
        subset = data[data["dataset"] == dataset]
        if dataset == "agrawal_gradual":
            ax.axvspan(80_000, 120_000, color="#fbbf24", alpha=0.18)
            ax.text(81_000, 1.23, "drift transition", fontsize=9, color="#92400e")
        else:
            ax.axvline(100_000, color="#111827", linestyle="--", linewidth=1.3)
            ax.text(101_000, 1.23, "drift point", fontsize=9, color="#111827")

        sns.lineplot(
            data=subset,
            x="instance",
            y="smooth_logloss",
            hue="method",
            hue_order=hue_order,
            palette=PALETTE,
            linewidth=2.0,
            ax=ax,
            legend=False,
        )
        if dataset == "agrawal_gradual":
            ax.axvline(100_000, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(titles[dataset], fontsize=13, weight="bold")
        ax.set_xlabel("Instance")
        ax.set_ylabel("Windowed 1k Log Loss")
        ax.grid(alpha=0.25)
        ax.ticklabel_format(style="plain", axis="x")
        sns.despine(ax=ax)

    fig.suptitle("Adaptation around controlled Agrawal drift", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.legend(
        handles=method_handles(hue_order),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.18)
    fig.tight_layout()
    savefig(fig, out_dir, "eval_synthetic_adaptation.png")


def plot_efficiency_tradeoff(summary: pd.DataFrame, out_dir: Path) -> None:
    data = keep_methods(summary[summary["dataset"].isin(REAL_DATASETS)], REAL_METHODS)
    data = data.copy()
    data["dataset_label"] = data["dataset"].str.capitalize()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, dataset in zip(axes, REAL_DATASETS):
        subset = data[data["dataset"] == dataset]
        sns.scatterplot(
            data=subset,
            x="elapsed_s",
            y="final_LogLoss",
            hue="method",
            hue_order=ordered_methods(data),
            palette=PALETTE,
            s=120,
            edgecolor="white",
            linewidth=0.8,
            ax=ax,
            legend=False,
        )
        for _, row in subset.iterrows():
            label = row["method"].replace(" drift-aware SRP", "").replace(" static top-k", " static")
            ax.annotate(
                label,
                (row["elapsed_s"], row["final_LogLoss"]),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_title(dataset.capitalize(), fontsize=13, weight="bold")
        ax.set_xlabel("Runtime, seconds (log scale)")
        ax.set_ylabel("Final Log Loss")
        ax.grid(alpha=0.25)
        sns.despine(ax=ax)

    fig.suptitle("Quality/runtime tradeoff on real CTR streams", x=0.02, ha="left", fontsize=15, weight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "eval_real_efficiency_tradeoff.png")


def format_seconds(value: float) -> str:
    if value >= 60:
        return f"{value / 60:.1f}m"
    if value >= 10:
        return f"{value:.0f}s"
    return f"{value:.1f}s"


def annotate_runtime_bars(ax: plt.Axes) -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append("" if pd.isna(value) else format_seconds(float(value)))
        ax.bar_label(container, labels=labels, fontsize=7, padding=2, rotation=90)


def plot_runtime_comparison(summary: pd.DataFrame, out_dir: Path) -> None:
    data = pd.concat(
        [
            keep_methods(summary[summary["dataset"].isin(REAL_DATASETS)], REAL_METHODS),
            keep_methods(summary[summary["dataset"].isin(SYNTHETIC_DATASETS)], SYNTHETIC_METHODS),
        ],
        ignore_index=True,
    )
    data["dataset"] = pd.Categorical(
        data["dataset"],
        ["avazu", "criteo", "agrawal_sudden", "agrawal_gradual"],
        ordered=True,
    )
    hue_order = ordered_methods(data)

    fig, ax = plt.subplots(figsize=(13.0, 6.3))
    sns.barplot(
        data=data,
        x="dataset",
        y="elapsed_s",
        hue="method",
        hue_order=hue_order,
        palette=PALETTE,
        ax=ax,
    )
    annotate_runtime_bars(ax)
    ax.set_yscale("log")
    ax.set_ylim(top=data["elapsed_s"].max() * 2.0)
    ax.set_title("Runtime comparison for 200k-stream evaluation", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Elapsed time, seconds (log scale)")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="", ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.27)
    savefig(fig, out_dir, "eval_runtime_comparison.png")


def plot_real_speedup(summary: pd.DataFrame, out_dir: Path) -> None:
    data = keep_methods(summary[summary["dataset"].isin(REAL_DATASETS)], REAL_METHODS).copy()
    baselines = (
        data[data["method"] == "DASRP drift-aware SRP"]
        .set_index("dataset")["elapsed_s"]
        .rename("dasrp_elapsed_s")
    )
    data = data.join(baselines, on="dataset")
    data["runtime_ratio_vs_dasrp"] = data["elapsed_s"] / data["dasrp_elapsed_s"]
    data = data[data["method"] != "DASRP drift-aware SRP"].copy()
    hue_order = [m for m in ordered_methods(data) if m in set(data["method"])]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.barplot(
        data=data,
        x="dataset",
        y="runtime_ratio_vs_dasrp",
        hue="method",
        hue_order=hue_order,
        palette=PALETTE,
        ax=ax,
    )
    for container in ax.containers:
        labels = [f"{value:.1f}x" for value in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title("Runtime ratio against DASRP on real CTR streams", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Runtime / DASRP runtime")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="", ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.25)
    savefig(fig, out_dir, "eval_real_speedup_vs_dasrp.png")


def plot_drift_counts(summary: pd.DataFrame, out_dir: Path) -> None:
    data = pd.concat(
        [
            keep_methods(summary[summary["dataset"].isin(REAL_DATASETS)], REAL_METHODS),
            keep_methods(summary[summary["dataset"].isin(SYNTHETIC_DATASETS)], SYNTHETIC_METHODS),
        ],
        ignore_index=True,
    )
    hue_order = ordered_methods(data)

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    sns.barplot(
        data=data,
        x="dataset",
        y="drifts",
        hue="method",
        hue_order=hue_order,
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("ADWIN drift alarms by dataset and method", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Detected drifts")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="", ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    sns.despine(ax=ax)
    fig.subplots_adjust(bottom=0.26)
    savefig(fig, out_dir, "eval_drift_counts.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation plots from experiment CSVs.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latex") / "presentation" / "results" / "evaluation",
    )
    args = parser.parse_args()

    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        },
    )

    summary, long, _ = load_results(args.results_dir)
    plot_real_final_logloss(summary, args.out_dir)
    plot_real_logloss_trajectories(long, args.out_dir)
    plot_synthetic_adaptation(long, args.out_dir)
    plot_efficiency_tradeoff(summary, args.out_dir)
    plot_runtime_comparison(summary, args.out_dir)
    plot_real_speedup(summary, args.out_dir)
    plot_drift_counts(summary, args.out_dir)

    generated = sorted(p.name for p in args.out_dir.glob("eval_*.png"))
    print(f"Wrote {len(generated)} plots to {args.out_dir.resolve()}:")
    for name in generated:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
