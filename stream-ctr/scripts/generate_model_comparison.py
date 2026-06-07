#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "source",
    "model",
    "selector",
    "instances",
    "drifts",
    "final_Accuracy",
    "final_LogLoss",
    "final_AUC",
    "final_Windowed[1000]-Accuracy",
    "final_Windowed[1000]-LogLoss",
]


MODEL_LEVEL_METHODS = {
    ("baseline", "HT", "none"): "HT",
    ("baseline", "HAT", "none"): "HAT",
    ("baseline", "SRP", "none"): "SRP",
    ("driftaware", "HT", "drift_aware_selector"): "HT_DA",
    ("driftaware", "HAT", "drift_aware_selector"): "HAT_DA",
    ("driftaware", "DASRP", "drift_aware_srp"): "DASRP",
}


METHOD_ORDER = {
    "HT": 0,
    "HAT": 1,
    "SRP": 2,
    "HT_DA": 3,
    "HAT_DA": 4,
    "DASRP": 5,
}


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric(row, name):
    value = row.get(name, "")
    if value == "":
        return None
    return float(value)


def comparison_rows(results_dir):
    rows = []
    for source, filename in [
        ("baseline", "results_summary.csv"),
        ("driftaware", "driftaware_summary.csv"),
    ]:
        for row in read_csv(results_dir / filename):
            key = (source, row["model"], row["selector"])
            method = MODEL_LEVEL_METHODS.get(key)
            if method is None:
                continue
            out = {col: row.get(col, "") for col in SUMMARY_COLUMNS}
            out["method"] = method
            out["source"] = source
            rows.append(out)
    rows.sort(key=lambda r: (r["dataset"], METHOD_ORDER.get(r["method"], 99)))
    return rows


def ht_baseline_rows(results_dir):
    rows = []
    for row in read_csv(results_dir / "results_summary.csv"):
        if row.get("model") == "HT" and row.get("selector") == "none":
            out = {col: row.get(col, "") for col in SUMMARY_COLUMNS}
            out["method"] = "HT"
            out["source"] = "baseline"
            rows.append(out)
    rows.sort(key=lambda r: r["dataset"])
    return rows


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(row, col):
    value = row.get(col, "")
    if value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value


def write_markdown(path, rows, ht_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r["dataset"] for r in rows})
    lines = [
        "# Model Comparison",
        "",
        "HT is included as the standalone baseline method: `model=HT`, `selector=none`.",
        "The model-level comparison uses `HT`, `HAT`, `SRP`, `HT_DA`, `HAT_DA`, and `DASRP`.",
        "",
        "## HT Baseline",
        "",
        "| Dataset | Accuracy | LogLoss | AUC | Windowed Accuracy | Windowed LogLoss | Drifts |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ht_rows:
        lines.append(
            "| {dataset} | {acc} | {logloss} | {auc} | {wacc} | {wlogloss} | {drifts} |".format(
                dataset=row["dataset"],
                acc=fmt(row, "final_Accuracy"),
                logloss=fmt(row, "final_LogLoss"),
                auc=fmt(row, "final_AUC"),
                wacc=fmt(row, "final_Windowed[1000]-Accuracy"),
                wlogloss=fmt(row, "final_Windowed[1000]-LogLoss"),
                drifts=row.get("drifts", ""),
            )
        )

    for dataset in datasets:
        lines.extend([
            "",
            f"## {dataset}",
            "",
            "| Method | Accuracy | LogLoss | AUC | Windowed Accuracy | Windowed LogLoss | Drifts |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in [r for r in rows if r["dataset"] == dataset]:
            lines.append(
                "| {method} | {acc} | {logloss} | {auc} | {wacc} | {wlogloss} | {drifts} |".format(
                    method=row["method"],
                    acc=fmt(row, "final_Accuracy"),
                    logloss=fmt(row, "final_LogLoss"),
                    auc=fmt(row, "final_AUC"),
                    wacc=fmt(row, "final_Windowed[1000]-Accuracy"),
                    wlogloss=fmt(row, "final_Windowed[1000]-LogLoss"),
                    drifts=row.get("drifts", ""),
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_long_rows(results_dir):
    out = []
    for source, filename in [
        ("baseline", "results_long.csv"),
        ("driftaware", "driftaware_long.csv"),
    ]:
        for row in read_csv(results_dir / filename):
            key = (source, row["model"], row["selector"])
            method = MODEL_LEVEL_METHODS.get(key)
            if method is None:
                continue
            row = dict(row)
            row["method"] = method
            row["source"] = source
            out.append(row)
    return out


def write_plots(results_dir, summary_rows, long_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping PNG plots: matplotlib unavailable ({exc})")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, filename, ylabel in [
        ("final_Accuracy", "model_comparison_accuracy.png", "Accuracy"),
        ("final_LogLoss", "model_comparison_logloss.png", "LogLoss"),
        ("final_AUC", "model_comparison_auc.png", "AUC"),
    ]:
        datasets = sorted({r["dataset"] for r in summary_rows})
        methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in summary_rows)]
        x = range(len(datasets))
        width = 0.12
        fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), 5))
        for i, method in enumerate(methods):
            vals = []
            for dataset in datasets:
                row = next((r for r in summary_rows
                            if r["dataset"] == dataset and r["method"] == method), None)
                vals.append(metric(row, metric_name) if row else None)
            offsets = [v + (i - len(methods) / 2) * width for v in x]
            ax.bar(offsets, [v if v is not None else 0.0 for v in vals],
                   width=width, label=method)
        ax.set_xticks(list(x))
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(ncols=min(3, len(methods)))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=160)
        plt.close(fig)

    for dataset in sorted({r["dataset"] for r in long_rows}):
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in METHOD_ORDER:
            rows = [r for r in long_rows if r["dataset"] == dataset and r["method"] == method]
            if not rows:
                continue
            rows.sort(key=lambda r: int(r["instance"]))
            ax.plot(
                [int(r["instance"]) for r in rows],
                [metric(r, "Windowed[1000]-LogLoss") for r in rows],
                label=method,
                linewidth=1.5,
            )
        ax.set_title(dataset)
        ax.set_xlabel("instance")
        ax.set_ylabel("Windowed[1000]-LogLoss")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{dataset}_windowed_logloss.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = comparison_rows(results_dir)
    ht_rows = ht_baseline_rows(results_dir)
    long_rows = load_long_rows(results_dir)

    write_csv(results_dir / "model_comparison_summary.csv", rows, SUMMARY_COLUMNS)
    write_csv(results_dir / "ht_baseline_summary.csv", ht_rows, SUMMARY_COLUMNS)
    write_csv(
        results_dir / "model_comparison_long.csv",
        long_rows,
        ["dataset", "method", "source", "model", "selector", "instance",
         "Accuracy", "LogLoss", "AUC", "Windowed[1000]-Accuracy",
         "Windowed[1000]-LogLoss", "drift_detected"],
    )
    write_markdown(results_dir / "model_comparison.md", rows, ht_rows)
    write_plots(results_dir, rows, long_rows)

    print(f"Wrote {results_dir / 'model_comparison_summary.csv'}")
    print(f"Wrote {results_dir / 'ht_baseline_summary.csv'}")
    print(f"Wrote {results_dir / 'model_comparison_long.csv'}")
    print(f"Wrote {results_dir / 'model_comparison.md'}")
    print(f"Wrote plots to {results_dir / 'plots'}")


if __name__ == "__main__":
    main()
