package stream.ensemble;

import com.yahoo.labs.samoa.instances.Instance;
import stream.drift.AdwinDriftDetector;
import stream.evaluation.AccuracyMetric;
import stream.evaluation.LogLossMetric;
import stream.evaluation.PrequentialEvaluator;
import stream.evaluation.WindowedMetric;
import stream.features.DriftAwareSelector;
import stream.model.HoeffdingTreeModel;
import stream.model.SrpModel;
import stream.model.StreamModel;
import stream.provider.AgrawalStreamProvider;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.io.IOException;
import java.nio.file.Path;

public class SmokeTestDriftAwareSrp {

    public static void main(String[] args) throws IOException {
        String arffPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_hashed_100.arff";

        System.out.println("=== Test 1: standalone training (no drift) on Agrawal ===");
        standalone();

        System.out.println();
        System.out.println("=== Test 2: Agrawal w/ drift @10k - SRP vs DriftAwareSRP ===");
        runAgrawalCompare();

        System.out.println();
        System.out.println("=== Test 3: Avazu 50k - SRP vs DriftAwareSRP ===");
        runAvazuCompare(arffPath);

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void standalone() {
        StreamProvider sp = new AgrawalStreamProvider(2_000, 2_000, 1, 1);
        DriftAwareSrpModel m = new DriftAwareSrpModel(
                "DASRP", 5, 4, 7L, HoeffdingTreeModel::new);
        m.initialize(sp.getHeader());
        int n = 0;
        double minP = 1, maxP = 0;
        while (sp.hasNext()) {
            Instance i = sp.next();
            double p = m.predictProbability(i);
            if (p < minP) minP = p;
            if (p > maxP) maxP = p;
            m.train(i);
            n++;
        }
        System.out.printf("n=%d minP=%.4f maxP=%.4f weights=%s%n",
                n, minP, maxP, java.util.Arrays.toString(m.weightManager().weightsSnapshot()));
    }

    private static void runAgrawalCompare() throws IOException {
        long limit = 20_000;
        runOne("agrawal_srp",
                new SrpModel("SRP", 10, 60, 1),
                new AgrawalStreamProvider(20_000, 10_000, 1, 3),
                limit, false, null);
        runOne("agrawal_dasrp",
                new DriftAwareSrpModel("DASRP", 10, 5, 1L, HoeffdingTreeModel::new),
                new AgrawalStreamProvider(20_000, 10_000, 1, 3),
                limit, true, 5);
    }

    private static void runAvazuCompare(String arffPath) throws IOException {
        long limit = 50_000;
        runOne("avazu_srp",
                new SrpModel("SRP", 10, 30, 1),
                new ArffStreamProvider(arffPath),
                limit, false, null);
        runOne("avazu_dasrp",
                new DriftAwareSrpModel("DASRP", 10, 20, 1L, HoeffdingTreeModel::new),
                new ArffStreamProvider(arffPath),
                limit, true, 20);
    }

    private static void runOne(String tag,
                               StreamModel model,
                               StreamProvider provider,
                               long limit,
                               boolean withDriftAware,
                               Integer topK) throws IOException {
        AdwinDriftDetector adwin = new AdwinDriftDetector(0.002);

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new WindowedMetric(1_000, AccuracyMetric::new))
                .addMetric(new WindowedMetric(1_000, LogLossMetric::new))
                .logInterval(1_000)
                .withDriftDetector(adwin);

        if (withDriftAware && model instanceof DriftAwareSrpModel dasrp) {
            DriftAwareSelector selector = new DriftAwareSelector(topK, 1_500, 0.005);
            selector.initialize(provider.getHeader());
            selector.withListener((idx, newHeader, sel, removed, added, delta) -> {
                System.out.println("  [adapt] @" + idx
                        + " removed=" + removed + " added=" + added);
                dasrp.onDriftDetected(removed, added);
            });

            StreamProvider observingProvider = new ObservingProvider(provider, selector);

            ev.withDriftHandler((idx, inst, m) -> {
                System.out.println("  [drift] @" + idx);
                selector.onDriftDetected();
            });

            long t0 = System.currentTimeMillis();
            long n = ev.run(model, observingProvider, limit);
            long t1 = System.currentTimeMillis();

            printAndExport(tag, n, ev, t1 - t0);
        } else {
            ev.withDriftHandler((idx, inst, m) ->
                    System.out.println("  [drift] @" + idx));

            long t0 = System.currentTimeMillis();
            long n = ev.run(model, provider, limit);
            long t1 = System.currentTimeMillis();

            printAndExport(tag, n, ev, t1 - t0);
        }
    }

    private static void printAndExport(String tag, long n,
                                       PrequentialEvaluator ev, long ms) throws IOException {
        System.out.printf("[%s] n=%d acc=%.4f logloss=%.4f wAcc=%.4f wLL=%.4f time=%d ms drifts=%d%n",
                tag, n,
                ev.metrics().get(0).getValue(),
                ev.metrics().get(1).getValue(),
                ev.metrics().get(2).getValue(),
                ev.metrics().get(3).getValue(),
                ms, ev.driftCount());
        Path csv = Path.of("results", tag + ".csv");
        ev.exportCsv(csv);
        System.out.println("  CSV: " + csv);
    }

    private static class ObservingProvider implements StreamProvider {
        private final StreamProvider inner;
        private final stream.features.DriftAwareSelector selector;
        ObservingProvider(StreamProvider inner, stream.features.DriftAwareSelector selector) {
            this.inner = inner;
            this.selector = selector;
        }
        @Override public com.yahoo.labs.samoa.instances.Instances getHeader() {
            return inner.getHeader();
        }
        @Override public boolean hasNext() { return inner.hasNext(); }
        @Override public Instance next() {
            Instance ins = inner.next();
            selector.observe(ins, (int) ins.classValue());
            return ins;
        }
        @Override public void restart() { inner.restart(); }
    }
}