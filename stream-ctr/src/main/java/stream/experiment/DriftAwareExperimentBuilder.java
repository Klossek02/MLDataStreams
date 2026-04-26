package stream.experiment;

import stream.drift.AdwinDriftDetector;
import stream.ensemble.DriftAwareSrpModel;
import stream.evaluation.PrequentialEvaluator;
import stream.features.DriftAwareSelector;
import stream.model.HoeffdingTreeModel;
import stream.model.StreamModel;
import stream.provider.AgrawalStreamProvider;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public class DriftAwareExperimentBuilder {

    public record Spec(String datasetName,
                       Supplier<StreamProvider> streamFactory,
                       long limit,
                       int topK,
                       int windowSize,
                       double changeThreshold,
                       int ensembleSize,
                       int subspaceSize,
                       double adwinDelta,
                       int minWeakOverlap,
                       double weightResetValue,
                       double weightDecayRate,
                       double resetRatio,
                       int cooldownInstances,
                       int warmupInstances,
                       String variant) {}

    public static ExperimentResult runDriftAwareSelectorWithModel(
            Spec spec,
            String modelTag,
            Supplier<StreamModel> baseModelFactory) {

        StreamProvider raw = spec.streamFactory().get();
        StreamModel model = baseModelFactory.get();

        DriftAwareSelector selector = new DriftAwareSelector(
                spec.topK(), spec.windowSize(), spec.changeThreshold());
        AdwinDriftDetector adwin = new AdwinDriftDetector(spec.adwinDelta());
        final long warmup = spec.warmupInstances();

        selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
            System.out.println("  [adapt] @" + idx
                    + " removed=" + removed + " added=" + added);
            model.reset();
            model.initialize(newHeader);
        });

        String tag = spec.datasetName() + " / " + modelTag + " / DA-Selector";
        PrequentialEvaluator ev = new PrequentialEvaluator()
                .logInterval(1_000)
                .progressInterval(chooseProgressInterval(spec.limit()))
                .progressTag(tag)
                .withFeatureSelector(selector)
                .withDriftDetector(adwin)
                .withDriftHandler((idx, inst, m) -> {
                    if (idx < warmup) {
                        return;
                    }
                    System.out.println("  [drift] @" + idx);
                    selector.onDriftDetected();
                });
        for (var mf : ExperimentMatrix.defaultMetrics(1_000)) {
            ev.addMetric(mf.get());
        }

        long t0 = System.currentTimeMillis();
        long n = ev.run(model, raw, spec.limit());
        long elapsed = System.currentTimeMillis() - t0;

        return packResult(spec.datasetName(), modelTag,
                "drift_aware_selector", n, elapsed, ev);
    }

    public static ExperimentResult runDriftAwareSrp(Spec spec) {
        StreamProvider raw = spec.streamFactory().get();

        DriftAwareSrpModel dasrp = new DriftAwareSrpModel(
                "DASRP", spec.ensembleSize(), spec.subspaceSize(),
                42L, HoeffdingTreeModel::new)
                .withMinWeakOverlap(spec.minWeakOverlap())
                .withWeightRecovery(spec.weightResetValue(), spec.weightDecayRate())
                .withResetRatio(spec.resetRatio())
                .withAdaptationCooldown(spec.cooldownInstances());

        DriftAwareSelector selector = new DriftAwareSelector(
                spec.topK(), spec.windowSize(), spec.changeThreshold());
        selector.initialize(raw.getHeader());

        AdwinDriftDetector adwin = new AdwinDriftDetector(spec.adwinDelta());
        final long warmup = spec.warmupInstances();

        selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
            System.out.println("  [adapt] @" + idx
                    + " removed=" + removed + " added=" + added);
            dasrp.onDriftDetected(removed, added);
        });

        ObservingProvider obs = new ObservingProvider(raw, selector);

        String tag = spec.datasetName() + " / DASRP / DA-SRP";
        PrequentialEvaluator ev = new PrequentialEvaluator()
                .logInterval(1_000)
                .progressInterval(chooseProgressInterval(spec.limit()))
                .progressTag(tag)
                .withDriftDetector(adwin)
                .withDriftHandler((idx, inst, m) -> {
                    if (idx < warmup) {
                        return;
                    }
                    System.out.println("  [drift] @" + idx);
                    selector.onDriftDetected();
                });
        for (var mf : ExperimentMatrix.defaultMetrics(1_000)) {
            ev.addMetric(mf.get());
        }

        long t0 = System.currentTimeMillis();
        long n = ev.run(dasrp, obs, spec.limit());
        long elapsed = System.currentTimeMillis() - t0;

        return packResult(spec.datasetName(), "DASRP",
                "drift_aware_srp", n, elapsed, ev);
    }

    private static int chooseProgressInterval(long limit) {
        if (limit <= 0)         return 50_000;
        if (limit <= 30_000)    return 2_000;
        if (limit <= 200_000)   return 10_000;
        if (limit <= 1_000_000) return 50_000;
        return 100_000;
    }

    private static ExperimentResult packResult(String dataset, String modelTag,
                                               String selectorTag, long n, long elapsed,
                                               PrequentialEvaluator ev) {
        Map<String, Double> finalMetrics = new LinkedHashMap<>();
        for (var m : ev.metrics()) finalMetrics.put(m.getName(), m.getValue());
        String name = (dataset + "_" + modelTag + "_" + selectorTag).toLowerCase();
        return new ExperimentResult(
                name, dataset, modelTag, selectorTag,
                n, elapsed, finalMetrics,
                new ArrayList<>(ev.history()),
                new ArrayList<>(ev.driftEvents()));
    }

    public static List<Spec> defaultSpecs(String avazuPath, String criteoPath) {
        List<Spec> specs = new ArrayList<>();

        specs.add(new Spec("avazu",
                () -> new ArffStreamProvider(avazuPath),
                200_000L,
                20, 5_000, 0.005,
                10, 20,
                0.002,
                1,
                0.3, 0.001,
                0.5, 0,
                2_000,
                "default"));

        specs.add(new Spec("criteo",
                () -> new ArffStreamProvider(criteoPath),
                200_000L,
                20, 5_000, 0.005,
                10, 20,
                0.002,
                1,
                0.3, 0.001,
                0.5, 0,
                2_000,
                "default"));

        specs.add(new Spec("agrawal_sudden",
                () -> new AgrawalStreamProvider(200_000, 100_000, 1, 3),
                200_000L,
                5, 800, 0.05,
                10, 4,
                0.001,
                2,
                0.5, 0.005,
                0.5, 0,
                5_000,
                "default"));

        specs.add(new Spec("agrawal_gradual",
                () -> new AgrawalStreamProvider(200_000, 100_000, 40_000, 1, 3, 42),
                200_000L,
                5, 800, 0.05,
                10, 4,
                0.001,
                2,
                0.5, 0.005,
                0.5, 0,
                5_000,
                "default"));

        return specs;
    }
}