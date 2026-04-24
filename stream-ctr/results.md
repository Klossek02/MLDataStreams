# Experiments — Results

This directory contains all experimental results from prequential evaluation
of streaming classifiers on Click-Through Rate (CTR) prediction datasets and
synthetic drift benchmarks.

---

## 1. Experimental Setup

All experiments were executed on the **full** instance count of each dataset
(no truncation). For each (dataset, model, feature selector) combination,
prequential evaluation was performed: each instance is first **predicted**,
then the prediction is scored, and finally the model is **trained** on the
ground-truth label (test-then-train protocol).

### 1.1 Datasets

| Name              | Type       | Source                                | Instances | Features | Drift                                  |
|-------------------|------------|---------------------------------------|----------:|---------:|----------------------------------------|
| `avazu`           | Real CTR   | Avazu (hashed numeric features, ARFF) |     full  |  100 + 1 | Natural (unknown) drift                |
| `criteo`          | Real CTR   | Criteo (numeric features, ARFF)       |     full  |  100 + 1 | Natural (unknown) drift                |
| `agrawal_sudden`  | Synthetic  | MOA `AgrawalGenerator`                |    20 000 |    9 + 1 | Sudden drift @ 10 000 (fn 1 → 3)       |
| `agrawal_gradual` | Synthetic  | MOA `AgrawalGenerator`                |    20 000 |    9 + 1 | Gradual drift, width 4 000 around 10k  |

### 1.2 Models

| Tag    | Model                                     | Notes                                              |
|--------|-------------------------------------------|----------------------------------------------------|
| `HT`   | `HoeffdingTree` (MOA)                     | Single Hoeffding tree, baseline                    |
| `HAT`  | `HoeffdingAdaptiveTree` (MOA)             | Hoeffding tree with built-in per-node ADWIN        |
| `SRP`  | `StreamingRandomPatches` (MOA)            | Ensemble of 10 trees, 60% subspace per learner     |
| `DASRP`| `DriftAwareSrpModel` (custom)             | Ensemble with subspace adaptation + weighted vote  |

### 1.3 Feature Selectors

| Tag                   | Strategy                                                         |
|-----------------------|------------------------------------------------------------------|
| `none`                | All input features used (no selection)                           |
| `static_topk`         | Top-K features by Information Gain on a 5 000-instance warm-up   |
| `online_ranking`      | Top-K features re-ranked every 5 000 instances on a sliding 5k window |
| `drift_aware_selector`| Top-K features re-ranked **only after ADWIN-detected drift**, comparing IG before vs. after drift; model is reset and re-initialized on the new feature subset |
| `drift_aware_srp`     | Used internally by `DASRP`. Detects drift, identifies weakened/strengthened features, replaces them in per-model subspaces and **resets only affected ensemble members** (others retain learned knowledge) |

### 1.4 Common Configuration

- **Drift detector:** `ADWIN(delta = 0.002)` on 0/1 prediction error. For Agrawal datasets `delta = 0.001` (faster detection).
- **Warm-up (drift-aware variants only):** ADWIN alarms ignored during the first 2 000 (real CTR) / 5 000 (Agrawal) instances to avoid false alarms while the model is still stabilizing.
- **Snapshot interval:** every 1 000 instances (windowed metrics, history, CSV rows).
- **Random seed:** 42 (DASRP), 1 (SRP) — for reproducibility.

---

## 2. Experiment Matrix

Two independent runners produce the full matrix:

### 2.1 Baseline matrix — `RunExperiments` (36 experiments)

`4 datasets × 3 models (HT, HAT, SRP) × 3 selectors (none, static_topk, online_ranking)`

This is the **control** matrix — standard streaming classifiers paired with common feature-selection strategies.

### 2.2 Drift-aware matrix — `RunDriftAwareExperiments` (12 experiments)

`4 datasets × 3 variants:`

- `<dataset> + HT  + drift_aware_selector`
- `<dataset> + HAT + drift_aware_selector`
- `<dataset> + DASRP (with drift_aware_srp internally)`

This is the **proposed method** — drift-triggered feature adaptation, evaluated against the baselines.

**Total: 48 experiments per full run.**

---

## 3. Metrics

For each experiment we compute the following metrics, accumulated and snapshotted every 1 000 instances:

| Metric                          | Definition                                                       | Purpose                                              |
|---------------------------------|------------------------------------------------------------------|------------------------------------------------------|
| `Accuracy`                      | Cumulative fraction of correct predictions                       | Overall quality                                      |
| `LogLoss`                       | Mean of `−[y·log(p) + (1−y)·log(1−p)]`                           | Probability-aware loss (penalizes overconfidence)    |
| `AUC`                           | ROC AUC over the last 10 000 predictions (Mann-Whitney with tie averaging) | Threshold-independent, robust to class imbalance     |
| `Windowed[1000]-Accuracy`       | Accuracy over the last 1 000 predictions                         | Captures recovery dynamics around drift              |
| `Windowed[1000]-LogLoss`        | Log-loss over the last 1 000 predictions                         | Same, probability-aware                              |

Additionally recorded per experiment:

- `instances` — number of processed instances
- `elapsed_ms` — total wall-clock time
- `drifts` — number of drift events detected by ADWIN

---

## 4. Output Files

Six CSV files are written to `results/` after a full run.

### 4.1 From `RunExperiments` (baselines)

#### `results_long.csv` — long-format snapshots

One row per (experiment × snapshot bucket).

```
dataset, model, selector, instance, Accuracy, LogLoss, AUC, Windowed[1000]-Accuracy, Windowed[1000]-LogLoss, drift_detected
```

- `instance` — number of instances processed at this snapshot (e.g. 1000, 2000, …)
- `drift_detected` — `true` if any ADWIN drift event occurred near this bucket
- Suitable for time-series plots in pandas/seaborn:

```python
sns.lineplot(data=df, x="instance", y="Windowed[1000]-LogLoss",
             hue="model", style="selector", col="dataset")
```

#### `results_summary.csv` — final metrics per experiment

One row per experiment.

```
dataset, model, selector, instances, elapsed_ms, drifts,
final_Accuracy, final_LogLoss, final_AUC,
final_Windowed[1000]-Accuracy, final_Windowed[1000]-LogLoss
```

#### `results_drifts.csv` — all drift events

One row per detected drift.

```
dataset, model, selector, instance, timestamp, windowBefore, windowAfter, detector
```

- `windowBefore` / `windowAfter` — ADWIN window length immediately before and after the cut
- Useful for overlaying vertical drift markers on plots

### 4.2 From `RunDriftAwareExperiments` (proposed method)

The schema is identical to the baseline files — only the file names differ:

| File                    | Equivalent of        | Contains                                      |
|-------------------------|----------------------|-----------------------------------------------|
| `driftaware_long.csv`   | `results_long.csv`   | 12 drift-aware experiments, snapshot rows      |
| `driftaware_summary.csv`| `results_summary.csv`| 12 drift-aware experiments, final metrics      |
| `driftaware_drifts.csv` | `results_drifts.csv` | Drift events from drift-aware experiments      |

In the `selector` column you will see `drift_aware_selector` or `drift_aware_srp`. In the `model` column for DASRP rows you will see `DASRP`.

---

## 5. Reproducing the Results

From the project root:

```bash
# Compile
mvn -q compile

# Run the full pipeline (baselines + drift-aware)
mvn -q exec:java -Dexec.mainClass=stream.Main \
    -Dexec.args="--run-all"
```

Or run the two phases separately:

```bash
mvn -q exec:java -Dexec.mainClass=stream.experiment.RunExperiments

mvn -q exec:java -Dexec.mainClass=stream.experiment.RunDriftAwareExperiments
```

After completion, the six CSV files described in Section 4 are available in `results/` and can be analyzed with any data-science toolchain (pandas, matplotlib, seaborn, R, Excel, etc.).
