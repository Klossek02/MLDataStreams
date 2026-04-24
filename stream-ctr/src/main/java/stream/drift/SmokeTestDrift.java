package stream.drift;

import stream.evaluation.AccuracyMetric;
import stream.evaluation.LogLossMetric;
import stream.evaluation.PrequentialEvaluator;
import stream.evaluation.WindowedMetric;
import stream.model.HoeffdingTreeModel;
import stream.provider.AgrawalStreamProvider;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.io.IOException;
import java.nio.file.Path;

public class SmokeTestDrift {

    public static void main(String[] args) throws IOException {
        String arffPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_hashed_100.arff";

        System.out.println("=== Test 1: ADWIN bare-bones (synthetic shift) ===");
        bareAdwinTest();

        System.out.println();
        System.out.println("=== Test 2: ADWIN inside PrequentialEvaluator on Agrawal ===");
        runOnAgrawal();

        System.out.println();
        System.out.println("=== Test 3: ADWIN inside PrequentialEvaluator on Avazu ===");
        runOnAvazu(arffPath);

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void bareAdwinTest() {
        AdwinDriftDetector d = new AdwinDriftDetector(0.002);
        java.util.Random r = new java.util.Random(42);
        int driftsBefore = 0;
        int driftsAfter = 0;
        for (int i = 0; i < 5_000; i++) {
            if (d.detect(r.nextDouble() < 0.05 ? 1.0 : 0.0)) driftsBefore++;
        }
        int wMid = d.getWindowSize();
        for (int i = 0; i < 5_000; i++) {
            if (d.detect(r.nextDouble() < 0.50 ? 1.0 : 0.0)) driftsAfter++;
        }
        System.out.printf("drifts before shift=%d  after shift=%d  windowMid=%d  windowEnd=%d%n",
                driftsBefore, driftsAfter, wMid, d.getWindowSize());
        if (driftsAfter == 0) {
            throw new AssertionError("ADWIN should detect at least one drift after error rate jump");
        }
    }

    private static void runOnAgrawal() throws IOException {
        StreamProvider provider = new AgrawalStreamProvider(20_000, 10_000, 1, 3);
        AdwinDriftDetector adwin = new AdwinDriftDetector(0.002);

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new WindowedMetric(1_000, AccuracyMetric::new))
                .logInterval(1_000)
                .withDriftDetector(adwin)
                .withDriftHandler((idx, inst, m) ->
                        System.out.println("  [drift] @" + idx
                                + " window=" + adwin.getWindowSize()));

        long n = ev.run(new HoeffdingTreeModel(), provider, 20_000);

        System.out.printf("Processed %d  drifts=%d  warnings=%d%n",
                n, ev.driftCount(), ev.warningCount());
        for (DriftEvent de : ev.driftEvents()) {
            System.out.printf("  event @%d  wBefore=%d  wAfter=%d%n",
                    de.instanceIndex(), de.windowSizeBefore(), de.windowSizeAfter());
        }
        Path csv = Path.of("results", "agrawal_ht_adwin.csv");
        ev.exportCsv(csv);
        System.out.println("CSV: " + csv.toAbsolutePath());
    }

    private static void runOnAvazu(String arffPath) throws IOException {
        StreamProvider provider = new ArffStreamProvider(arffPath);
        AdwinDriftDetector adwin = new AdwinDriftDetector(0.002);

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new WindowedMetric(2_000, AccuracyMetric::new))
                .logInterval(2_000)
                .withDriftDetector(adwin)
                .withDriftHandler((idx, inst, m) ->
                        System.out.println("  [drift] @" + idx
                                + " window=" + adwin.getWindowSize()));

        long n = ev.run(new HoeffdingTreeModel(), provider, 50_000);

        System.out.printf("Processed %d  drifts=%d%n", n, ev.driftCount());
        Path csv = Path.of("results", "avazu_ht_adwin.csv");
        ev.exportCsv(csv);
        System.out.println("CSV: " + csv.toAbsolutePath()
                + "  events=" + ev.driftEvents().size());
    }
}