package stream.experiment;

import stream.drift.DriftDetector;
import stream.evaluation.Metric;
import stream.evaluation.PrequentialEvaluator;
import stream.features.FeatureSelector;
import stream.model.StreamModel;
import stream.provider.StreamProvider;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Supplier;

public class ExperimentRunner {

    public ExperimentResult run(Experiment exp) {
        long timeoutMin = chooseTimeoutMinutes(exp.datasetName(), exp.modelName());
        ExecutorService es = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "exp-" + exp.name());
            t.setDaemon(true);
            return t;
        });
        Future<ExperimentResult> fut = es.submit(() -> runUnsafe(exp));
        try {
            return fut.get(timeoutMin, TimeUnit.MINUTES);
        } catch (TimeoutException te) {
            fut.cancel(true);
            System.err.printf("  TIMEOUT after %d min: %s%n", timeoutMin, exp.name());
            return null;
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            return null;
        } catch (Exception e) {
            System.err.println("  FAILED: " + e.getMessage());
            e.printStackTrace();
            return null;
        } finally {
            es.shutdownNow();
        }
    }

    private ExperimentResult runUnsafe(Experiment exp) {
        StreamProvider provider = exp.streamFactory().get();
        StreamModel model = exp.modelFactory().get();

        int progressInterval = chooseProgressInterval(exp.limit());
        String tag = exp.datasetName() + " / " + exp.modelName() + " / " + exp.selectorName();

        PrequentialEvaluator ev = new PrequentialEvaluator()
                .logInterval(exp.logInterval())
                .progressInterval(progressInterval)
                .progressTag(tag);

        for (Supplier<Metric> mf : exp.metricFactories()) {
            ev.addMetric(mf.get());
        }
        if (exp.selectorFactory() != null) {
            FeatureSelector sel = exp.selectorFactory().get();
            ev.withFeatureSelector(sel);
        }
        if (exp.detectorFactory() != null) {
            DriftDetector dd = exp.detectorFactory().get();
            ev.withDriftDetector(dd);
        }

        long t0 = System.currentTimeMillis();
        long n = ev.run(model, provider, exp.limit());
        long elapsed = System.currentTimeMillis() - t0;

        Map<String, Double> finalMetrics = new LinkedHashMap<>();
        for (Metric m : ev.metrics()) {
            finalMetrics.put(m.getName(), m.getValue());
        }
        return new ExperimentResult(
                exp.name(), exp.datasetName(), exp.modelName(), exp.selectorName(),
                n, elapsed,
                finalMetrics,
                new ArrayList<>(ev.history()),
                new ArrayList<>(ev.driftEvents()));
    }

    public List<ExperimentResult> runAll(List<Experiment> experiments) {
        List<ExperimentResult> out = new ArrayList<>();
        for (int i = 0; i < experiments.size(); i++) {
            Experiment exp = experiments.get(i);
            System.out.printf("[%d/%d] running %s%n", i + 1, experiments.size(), exp.name());
            ExperimentResult r = run(exp);
            if (r != null) {
                System.out.printf("  done n=%d ms=%d %s%n",
                        r.instances(), r.elapsedMs(), r.finalMetrics());
                out.add(r);
            } else {
                System.out.printf("  skipped (no result): %s%n", exp.name());
            }
        }
        return out;
    }

    private static int chooseProgressInterval(long limit) {
        if (limit <= 0)            return 50_000;
        if (limit <= 30_000)       return 2_000;
        if (limit <= 200_000)      return 10_000;
        if (limit <= 1_000_000)    return 50_000;
        return 100_000;
    }

    private static long chooseTimeoutMinutes(String dataset, String model) {
        boolean isSrp = model.contains("SRP");
        if (dataset.startsWith("agrawal")) {
            return isSrp ? 15 : 5;
        }
        return isSrp ? 60 : 20;
    }
}