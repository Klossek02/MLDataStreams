package stream.experiment;

import stream.drift.AdwinDriftDetector;
import stream.evaluation.AccuracyMetric;
import stream.evaluation.AucMetric;
import stream.evaluation.LogLossMetric;
import stream.evaluation.Metric;
import stream.evaluation.WindowedMetric;
import stream.features.FeatureSelector;
import stream.features.NoSelector;
import stream.features.OnlineRankingSelector;
import stream.features.StaticTopKSelector;
import stream.model.HatModel;
import stream.model.HoeffdingTreeModel;
import stream.model.SrpModel;
import stream.model.StreamModel;
import stream.provider.AgrawalStreamProvider;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class ExperimentMatrix {

    public record Dataset(String name, Supplier<StreamProvider> factory, long limit) {}

    public record SelectorSpec(String name, Supplier<FeatureSelector> factory) {}

    public record ModelSpec(String name, Supplier<StreamModel> factory) {}

    public static List<Supplier<Metric>> defaultMetrics(int window) {
        return List.of(
                AccuracyMetric::new,
                LogLossMetric::new,
                () -> new AucMetric(10_000),
                () -> new WindowedMetric(window, AccuracyMetric::new),
                () -> new WindowedMetric(window, LogLossMetric::new)
        );
    }

    public static List<Experiment> build(String avazuPath, String criteoPath) {
        List<Dataset> datasets = List.of(
                new Dataset("avazu",
                        () -> new ArffStreamProvider(avazuPath), 200_000L),
                new Dataset("criteo",
                        () -> new ArffStreamProvider(criteoPath), 200_000L),
                new Dataset("agrawal_sudden",
                        () -> new AgrawalStreamProvider(200_000, 100_000, 1, 3), 200_000L),
                new Dataset("agrawal_gradual",
                        () -> new AgrawalStreamProvider(200_000, 100_000, 40_000, 1, 3, 42), 200_000L)
        );

        List<ModelSpec> models = List.of(
                new ModelSpec("HT", HoeffdingTreeModel::new),
                new ModelSpec("HAT", HatModel::new),
                new ModelSpec("SRP", () -> new SrpModel("SRP", 10, 60, 1))
        );

        List<Experiment> all = new ArrayList<>();
        for (Dataset d : datasets) {
            int featureCount = featureCountFor(d);
            int topK = Math.max(2, Math.min(20, featureCount / 5));
            int warmup = (int) Math.min(5_000, d.limit() / 4);

            List<SelectorSpec> selectors = List.of(
                    new SelectorSpec("none", NoSelector::new),
                    new SelectorSpec("static_topk",
                            () -> staticPreWarmed(d.factory(), topK, warmup)),
                    new SelectorSpec("online_ranking",
                            () -> new OnlineRankingSelector(topK, warmup, warmup))
            );

            for (ModelSpec m : models) {
                for (SelectorSpec s : selectors) {
                    all.add(Experiment.builder()
                            .dataset(d.name(), d.factory())
                            .model(m.name(), m.factory())
                            .selector(s.name(), s.factory())
                            .detector(() -> new AdwinDriftDetector(0.002))
                            .metrics(defaultMetrics(1_000))
                            .limit(d.limit())
                            .logInterval(1_000)
                            .build());
                }
            }
        }
        return all;
    }

    private static int featureCountFor(Dataset d) {
        StreamProvider sp = d.factory().get();
        return sp.getHeader().numAttributes() - 1;
    }

    private static FeatureSelector staticPreWarmed(Supplier<StreamProvider> factory,
                                                   int topK, int warmup) {
        StaticTopKSelector sel = new StaticTopKSelector(topK, warmup);
        StreamProvider sp = factory.get();
        sel.initialize(sp.getHeader());
        int n = 0;
        while (sp.hasNext() && n < warmup) {
            var ins = sp.next();
            sel.observe(ins, (int) ins.classValue());
            n++;
        }
        sel.finishWarmup();
        if (!sel.isReady()) {
            throw new IllegalStateException(
                    "StaticTopK warmup failed after " + n + " / " + warmup + " instances");
        }
        return sel;
    }
}
