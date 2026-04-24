package stream.features;

import stream.drift.AdwinDriftDetector;
import stream.evaluation.AccuracyMetric;
import stream.evaluation.LogLossMetric;
import stream.evaluation.PrequentialEvaluator;
import stream.evaluation.WindowedMetric;
import stream.model.HoeffdingTreeModel;
import stream.model.StreamModel;
import stream.provider.AgrawalStreamProvider;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.io.IOException;
import java.nio.file.Path;

public class SmokeTestDriftAware {

    public static void main(String[] args) throws IOException {
        String arffPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_hashed_100.arff";

        System.out.println("=== Test 1: DriftAware on Agrawal (drift @ 10k, fn 1->3) ===");
        runOnAgrawal();

        System.out.println();
        System.out.println("=== Test 2: DriftAware on Avazu (50k) ===");
        runOnAvazu(arffPath);

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void runOnAgrawal() throws IOException {
        StreamProvider provider = new AgrawalStreamProvider(20_000, 10_000, 1, 3);
        StreamModel model = new HoeffdingTreeModel();

        DriftAwareSelector selector = new DriftAwareSelector(5, 1_500, 0.01);
        AdwinDriftDetector adwin = new AdwinDriftDetector(0.002);

        selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
            System.out.println("  [adapt] @" + idx
                    + " removed=" + removed
                    + " added=" + added
                    + " selected=" + sel);
            model.reset();
            model.initialize(newHeader);
        });

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new WindowedMetric(1_000, AccuracyMetric::new))
                .logInterval(1_000)
                .withFeatureSelector(selector)
                .withDriftDetector(adwin)
                .withDriftHandler((idx, inst, m) -> {
                    System.out.println("  [drift] @" + idx
                            + " window=" + adwin.getWindowSize());
                    selector.onDriftDetected();
                });

        long n = ev.run(model, provider, 20_000);
        System.out.printf("Processed %d  drifts=%d  adaptations=%d%n",
                n, ev.driftCount(), selector.events().size());
        Path csv = Path.of("results", "agrawal_ht_driftaware.csv");
        ev.exportCsv(csv);
        System.out.println("CSV: " + csv);
    }

    private static void runOnAvazu(String arffPath) throws IOException {
        StreamProvider provider = new ArffStreamProvider(arffPath);
        StreamModel model = new HoeffdingTreeModel();

        DriftAwareSelector selector = new DriftAwareSelector(20, 5_000, 0.005);
        AdwinDriftDetector adwin = new AdwinDriftDetector(0.002);

        selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
            System.out.println("  [adapt] @" + idx
                    + " removed=" + removed
                    + " added=" + added);
            model.reset();
            model.initialize(newHeader);
        });

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new WindowedMetric(2_000, AccuracyMetric::new))
                .logInterval(2_000)
                .withFeatureSelector(selector)
                .withDriftDetector(adwin)
                .withDriftHandler((idx, inst, m) -> {
                    System.out.println("  [drift] @" + idx
                            + " window=" + adwin.getWindowSize());
                    selector.onDriftDetected();
                });

        long n = ev.run(model, provider, 50_000);
        System.out.printf("Processed %d  drifts=%d  adaptations=%d%n",
                n, ev.driftCount(), selector.events().size());
        Path csv = Path.of("results", "avazu_ht_driftaware.csv");
        ev.exportCsv(csv);
        System.out.println("CSV: " + csv);
    }
}