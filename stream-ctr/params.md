# Key Parameters — Where to Set and What They Control

## ExperimentMatrix.java — Main Configuration Point

### 1. Instance limits per dataset

```java
new Dataset("avazu",    () -> new ArffStreamProvider(avazuPath), 50_000L),   // ← here
new Dataset("criteo",   () -> new ArffStreamProvider(criteoPath), 50_000L),  // ← here
new Dataset("agrawal_sudden",  ..., 20_000L),                                // ← here
new Dataset("agrawal_gradual", ..., 20_000L),                                // ← here
```

**What it does:** how many instances to process from the stream. For a full run change to `1_000_000L` (Avazu/Criteo) and `100_000L` (Agrawal).

---

### 2. ADWIN delta (drift detection sensitivity)

```java
.detector(() -> new AdwinDriftDetector(0.002))   // ← here
```

**What it does:** smaller delta = fewer false alarms, slower detection. Larger = faster but noisier.

| Value | Behavior | When to use |
|-------|----------|-------------|
| 0.001 | Very conservative | Agrawal (few features, clear drift) |
| 0.002 | Default | Avazu, Criteo (many features, subtle drift) |
| 0.01 | Aggressive | Testing, fast prototyping |

The current code uses one value for all datasets. To differentiate per dataset:

```java
double delta = d.name().startsWith("agrawal") ? 0.001 : 0.002;
.detector(() -> new AdwinDriftDetector(delta))
```

---

### 3. Top-K (how many features to select)

```java
int topK = Math.max(2, Math.min(20, featureCount / 5));   // ← here
```

**What it does:** how many features to keep after selection. `featureCount / 5` = 20% of features.

| Dataset | featureCount | topK |
|---------|-------------|------|
| Avazu | 109 | 20 (capped) |
| Criteo | 128 | 20 (capped) |
| Agrawal | 9 | 2 (floor) |

To select more/fewer features, change `20` (upper cap) or `5` (divisor).

---

### 4. Warmup (instances before selection kicks in)

```java
int warmup = (int) Math.min(5_000, d.limit() / 4);   // ← here
```

**What it does:** how many instances to collect before StaticTopK selects features / before OnlineRanking starts re-ranking. Also used as the sliding window size in OnlineRanking.

| Dataset | limit | warmup |
|---------|-------|--------|
| Avazu 50k | 50 000 | 5 000 |
| Criteo 50k | 50 000 | 5 000 |
| Agrawal | 20 000 | 5 000 |
| Avazu 1M | 1 000 000 | 5 000 |

---

### 5. SRP — ensemble size and subspace

```java
new ModelSpec("SRP", () -> new SrpModel("SRP", 10, 60, 1))
//                                             ↑   ↑   ↑
//                                    ensembleSize  %   seed
```

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `10` | ensembleSize | Number of trees in the ensemble |
| `60` | subspacePercent | Each tree sees 60% of features |
| `1` | seed | Random seed |

More trees = better results, slower. 60% subspace is the standard in SRP literature.

---

### 6. Snapshot interval (logging frequency)

```java
.logInterval(1_000)   // ← here
```

**What it does:** how often to write a metric snapshot to CSV. On the full million, 1000 produces 1000 rows per experiment — dense enough for smooth plots. Increase to 5000 if CSVs get too large.

---

### 7. Windowed metric size

```java
() -> new WindowedMetric(window, AccuracyMetric::new)
//                        ↑
//                    defaultMetrics(1_000)  ← here (argument in build())
```

**What it does:** last N instances used to compute windowed accuracy/logloss. 1000 is a good trade-off — smooth enough but responsive to changes.

---

### 8. AUC buffer

```java
() -> new AucMetric(10_000)   // ← here
```

**What it does:** AUC computed over the last 10 000 instances (Mann-Whitney). Larger window = more stable but less sensitive to local changes.

---

## RunDriftAwareExperiments — Additional Parameters

These parameters are in `DriftAwareExperimentMatrix` or directly in constructors:

### 9. DriftAwareSelector — changeThreshold

```java
new DriftAwareSelector(topK, windowSize, changeThreshold)
//                             ↑              ↑
//                          e.g. 800        e.g. 0.01
```

| Parameter | Typical value | What it does |
|-----------|--------------|-------------|
| `windowSize` | 800 | Instances to collect after drift for window comparison |
| `changeThreshold` | 0.01 | Minimum IG change to consider a feature as changed |

Smaller `windowSize` = faster reaction but less data for comparison. Smaller `changeThreshold` = more features swapped per adaptation.

### 10. DriftAwareSrpModel — reset threshold

```java
// in SubspaceManager.adaptSubspaces():
double overlapRatio = (double) overlap.size() / sub.size();
if (overlapRatio < 0.5) { ... }   // ← here
```

**What it does:** if a model has fewer than 50% weak features in its subspace, swap the features but do NOT reset the model. Above 50% = full reset.

### 11. WeightManager — post-reset weights

```java
private final double resetWeight  = 0.3;    // ← weight right after reset
private final double normalWeight = 1.0;    // ← weight of a stable model
private final double recoveryRate = 0.001;  // ← weight increase per instance
```

`recoveryRate = 0.001` means a reset model recovers to full weight after ~700 instances.

### 12. Drift warmup (ignoring early alarms)

```java
int driftWarmup = d.name().startsWith("agrawal") ? 5_000 : 2_000;   // ← here
```

**What it does:** ignores ADWIN alarms for the first N instances. Prevents false alarms while the model is still unstable.

---

## Quick Reference — What to Change for a Full Run

```java
// ExperimentMatrix.build():
new Dataset("avazu",    ..., 1_000_000L),       // was 50_000L
new Dataset("criteo",   ..., 1_000_000L),       // was 50_000L
new Dataset("agrawal_sudden",  ..., 100_000L),  // was 20_000L
new Dataset("agrawal_gradual", ..., 100_000L),  // was 20_000L

// Agrawal driftPoint accordingly:
new AgrawalStreamProvider(100_000, 50_000, 1, 3)        // sudden
new AgrawalStreamProvider(100_000, 50_000, 20_000, ...) // gradual
```

All other parameters remain unchanged.
