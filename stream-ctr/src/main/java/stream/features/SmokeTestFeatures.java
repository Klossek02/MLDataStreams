package stream.features;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import stream.evaluation.AccuracyMetric;
import stream.evaluation.LogLossMetric;
import stream.evaluation.PrequentialEvaluator;
import stream.evaluation.WindowedMetric;
import stream.model.HoeffdingTreeModel;
import stream.model.StreamModel;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SmokeTestFeatures {

    public static void main(String[] args) throws IOException {
        String arffPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_hashed_100.arff";
        long limit = (args.length > 1) ? Long.parseLong(args[1]) : 30_000L;

        System.out.println("=== Test 1: InfoGain ranking on 5k Avazu warmup ===");
        rankingSanityCheck(arffPath);

        System.out.println();
        System.out.println("=== Test 2: NoSelector vs StaticTopK vs OnlineRanking on Avazu ===");
        compareSelectors(arffPath, limit);

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void rankingSanityCheck(String arffPath) {
        StreamProvider sp = new ArffStreamProvider(arffPath);
        Instances header = sp.getHeader();
        List<Instance> buf = new ArrayList<>();
        int n = 0;
        while (sp.hasNext() && n < 5_000) {
            buf.add(sp.next());
            n++;
        }
        InfoGainRanker r = new InfoGainRanker(10);
        Map<Integer, Double> ig = r.rank(header, buf);
        List<Map.Entry<Integer, Double>> top = new ArrayList<>(ig.entrySet());
        top.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        System.out.println("Top-10 features by IG:");
        for (int i = 0; i < Math.min(10, top.size()); i++) {
            int idx = top.get(i).getKey();
            System.out.printf("  [%3d] %-20s IG=%.6f%n",
                    idx, header.attribute(idx).name(), top.get(i).getValue());
        }
    }

    private static void compareSelectors(String arffPath, long limit) throws IOException {
        runWithSelector("none", arffPath, limit, new NoSelector());
        runWithSelector("static-top20", arffPath, limit,
                preWarmedStatic(arffPath, 20, 5_000));
        runWithSelector("online-top20", arffPath, limit,
                new OnlineRankingSelector(20, 5_000, 5_000));
    }

    private static StaticTopKSelector preWarmedStatic(String arffPath, int k, int warmup) {
        StaticTopKSelector sel = new StaticTopKSelector(k, warmup);
        StreamProvider sp = new ArffStreamProvider(arffPath);
        sel.initialize(sp.getHeader());
        int n = 0;
        while (sp.hasNext() && n < warmup) {
            Instance ins = sp.next();
            sel.observe(ins, (int) ins.classValue());
            n++;
        }
        if (!sel.isReady()) {
            throw new IllegalStateException("StaticTopK warmup failed");
        }
        System.out.println("Static pre-warmed selected: " + sel.getSelectedIndices());
        return sel;
    }

    private static void runWithSelector(String tag, String arffPath, long limit,
                                        FeatureSelector selector) throws IOException {
        StreamProvider provider = new ArffStreamProvider(arffPath);
        StreamModel model = new HoeffdingTreeModel();

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .addMetric(new AccuracyMetric())
                .addMetric(new LogLossMetric())
                .addMetric(new WindowedMetric(2_000, AccuracyMetric::new))
                .logInterval(2_000)
                .withFeatureSelector(selector);

        if (selector instanceof OnlineRankingSelector ors) {
            ors.withReinitListener((newHeader, newSet) -> {
                System.out.println("  [reinit] online selected=" + newSet);
                model.reset();
                model.initialize(newHeader);
            });
        }

        long t0 = System.currentTimeMillis();
        long n = ev.run(model, provider, limit);
        long t1 = System.currentTimeMillis();

        double acc = ev.metrics().get(0).getValue();
        double ll = ev.metrics().get(1).getValue();
        double wAcc = ev.metrics().get(2).getValue();
        System.out.printf("[%s] n=%d acc=%.4f logloss=%.4f wAcc=%.4f time=%d ms selected=%d%n",
                tag, n, acc, ll, wAcc, (t1 - t0),
                selector.getSelectedIndices() == null ? -1 : selector.getSelectedIndices().size());

        Path csv = Path.of("results", "avazu_ht_" + tag + ".csv");
        ev.exportCsv(csv);
        System.out.println("  CSV: " + csv);
    }
}