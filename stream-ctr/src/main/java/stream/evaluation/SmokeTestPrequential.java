package stream.evaluation;

import stream.model.HoeffdingTreeModel;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.io.IOException;
import java.nio.file.Path;

public class SmokeTestPrequential {

    public static void main(String[] args) throws IOException {
        String arffPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_hashed_100.arff";
        long limit = (args.length > 1) ? Long.parseLong(args[1]) : 50_000L;
        Path outCsv = Path.of("results", "avazu_ht_prequential.csv");

        StreamProvider provider = new ArffStreamProvider(arffPath);

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new AucMetric(10_000))
                .addMetric(new WindowedMetric(1_000, AccuracyMetric::new))
                .addMetric(new WindowedMetric(1_000, LogLossMetric::new))
                .logInterval(1_000)
                .withDriftHandler((idx, inst, m) ->
                        System.out.println("  [drift] @" + idx));

        long t0 = System.currentTimeMillis();
        long n = ev.run(new HoeffdingTreeModel(), provider, limit);
        long t1 = System.currentTimeMillis();

        System.out.printf("Processed %d instances in %d ms (drifts=%d warnings=%d)%n",
                n, (t1 - t0), ev.driftCount(), ev.warningCount());
        for (Metric m : ev.metrics()) {
            System.out.printf("  %-24s = %.4f%n", m.getName(), m.getValue());
        }

        ev.exportCsv(outCsv);
        System.out.println("CSV: " + outCsv.toAbsolutePath()
                + " (" + ev.history().size() + " rows)");

        EvalRecord first = ev.history().get(0);
        EvalRecord last = ev.history().get(ev.history().size() - 1);
        System.out.println("First snapshot: " + first.instanceIndex() + " " + first.values());
        System.out.println("Last  snapshot: " + last.instanceIndex() + " " + last.values());
    }
}