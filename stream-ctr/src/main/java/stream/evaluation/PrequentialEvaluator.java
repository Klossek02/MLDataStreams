package stream.evaluation;

import com.yahoo.labs.samoa.instances.Instance;
import stream.drift.DriftDetector;
import stream.drift.DriftEvent;
import stream.features.FeatureSelector;
import stream.model.StreamModel;
import stream.provider.StreamProvider;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class PrequentialEvaluator {

    public interface DriftHandler {
        void onDrift(long instanceIndex, Instance instance, StreamModel model);
    }

    private final List<Metric> metrics = new ArrayList<>();
    private final List<EvalRecord> history = new ArrayList<>();
    private final List<DriftEvent> driftEvents = new ArrayList<>();

    private FeatureSelector featureSelector;
    private DriftDetector driftDetector;
    private DriftHandler driftHandler;

    private int logInterval = 1_000;
    private boolean useLogLossForDrift = false;
    private long driftCount = 0;
    private long warningCount = 0;

    private int progressInterval = 10_000;
    private boolean progressEnabled = true;
    private String progressTag = "";

    public PrequentialEvaluator addMetric(Metric m) {
        metrics.add(m);
        return this;
    }

    public PrequentialEvaluator withFeatureSelector(FeatureSelector fs) {
        this.featureSelector = fs;
        return this;
    }

    public PrequentialEvaluator withDriftDetector(DriftDetector dd) {
        this.driftDetector = dd;
        return this;
    }

    public PrequentialEvaluator withDriftHandler(DriftHandler h) {
        this.driftHandler = h;
        return this;
    }

    public PrequentialEvaluator logInterval(int n) {
        this.logInterval = Math.max(1, n);
        return this;
    }

    public PrequentialEvaluator driftSignalLogLoss(boolean enable) {
        this.useLogLossForDrift = enable;
        return this;
    }

    public PrequentialEvaluator progressInterval(int n) {
        this.progressInterval = Math.max(1, n);
        return this;
    }

    public PrequentialEvaluator progressEnabled(boolean b) {
        this.progressEnabled = b;
        return this;
    }

    public PrequentialEvaluator progressTag(String tag) {
        this.progressTag = (tag == null) ? "" : tag;
        return this;
    }

    public List<Metric> metrics() {
        return metrics;
    }

    public List<EvalRecord> history() {
        return history;
    }

    public List<DriftEvent> driftEvents() {
        return driftEvents;
    }

    public long driftCount() {
        return driftCount;
    }

    public long warningCount() {
        return warningCount;
    }

    public long run(StreamModel model, StreamProvider provider, long limit) {
        if (featureSelector != null) {
            featureSelector.initialize(provider.getHeader());
            model.initialize(featureSelector.filteredHeader());
        } else {
            model.initialize(provider.getHeader());
        }
        for (Metric m : metrics) m.reset();
        if (driftDetector != null) driftDetector.reset();
        history.clear();
        driftEvents.clear();
        driftCount = 0;
        warningCount = 0;

        long startMs = System.currentTimeMillis();
        long t = 0;
        while (provider.hasNext() && (limit <= 0 || t < limit)) {
            Instance raw = provider.next();
            int y = (int) raw.classValue();

            Instance filtered = (featureSelector != null)
                    ? featureSelector.filter(raw)
                    : raw;

            double p = model.predictProbability(filtered);

            for (Metric m : metrics) m.update(p, y);

            if (driftDetector != null) {
                double signal = computeDriftSignal(p, y);
                int wBefore = driftDetector.getWindowSize();
                boolean drift = driftDetector.detect(signal);
                if (driftDetector.isWarning()) warningCount++;
                if (drift) {
                    driftCount++;
                    int wAfter = driftDetector.getWindowSize();
                    DriftEvent ev = new DriftEvent(
                            t,
                            System.currentTimeMillis(),
                            wBefore,
                            wAfter,
                            driftDetector.getName());
                    driftEvents.add(ev);
                    if (driftHandler != null) {
                        driftHandler.onDrift(t, raw, model);
                    }
                }
            }

            model.train(filtered);
            if (featureSelector != null) {
                featureSelector.observe(raw, y);
            }

            t++;
            if (t % logInterval == 0) {
                recordSnapshot(t);
            }
            if (progressEnabled && t % progressInterval == 0) {
                printProgress(t, limit, startMs);
            }
        }
        if (t > 0 && t % logInterval != 0) {
            recordSnapshot(t);
        }
        if (progressEnabled) {
            printProgress(t, limit, startMs);
        }
        return t;
    }

    private void printProgress(long t, long limit, long startMs) {
        long elapsedMs = System.currentTimeMillis() - startMs;
        double rate = elapsedMs > 0 ? (t * 1000.0 / elapsedMs) : 0.0;
        String tag = progressTag.isEmpty() ? "" : "[" + progressTag + "] ";
        if (limit > 0) {
            double pct = 100.0 * t / limit;
            long remainingMs = rate > 0 ? (long) ((limit - t) * 1000.0 / rate) : -1;
            System.out.printf(
                    "  %sprogress: %,d / %,d  (%5.1f%%)  rate=%.0f inst/s  elapsed=%ds  eta=%ds%n",
                    tag, t, limit, pct, rate, elapsedMs / 1000, Math.max(0, remainingMs / 1000));
        } else {
            System.out.printf(
                    "  %sprogress: %,d  rate=%.0f inst/s  elapsed=%ds%n",
                    tag, t, rate, elapsedMs / 1000);
        }
    }

    private double computeDriftSignal(double p, int y) {
        if (useLogLossForDrift) {
            double pc = Math.max(1e-15, Math.min(1.0 - 1e-15, p));
            return -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
        }
        int yHat = (p >= 0.5) ? 1 : 0;
        return (yHat == y) ? 0.0 : 1.0;
    }

    private void recordSnapshot(long t) {
        String[] names = new String[metrics.size()];
        double[] vals = new double[metrics.size()];
        for (int i = 0; i < metrics.size(); i++) {
            names[i] = metrics.get(i).getName();
            vals[i] = metrics.get(i).getValue();
        }
        history.add(EvalRecord.of(t, names, vals));
    }

    public void exportCsv(Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write("instance");
            for (Metric m : metrics) {
                w.write(",");
                w.write(escape(m.getName()));
            }
            w.newLine();
            for (EvalRecord r : history) {
                w.write(Long.toString(r.instanceIndex()));
                for (Metric m : metrics) {
                    Double v = r.values().get(m.getName());
                    w.write(",");
                    w.write(v == null ? "" : Double.toString(v));
                }
                w.newLine();
            }
        }
    }

    public void exportDriftEventsCsv(Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write("instance,timestamp,windowBefore,windowAfter,detector");
            w.newLine();
            for (DriftEvent e : driftEvents) {
                w.write(e.instanceIndex() + ","
                        + e.timestamp() + ","
                        + e.windowSizeBefore() + ","
                        + e.windowSizeAfter() + ","
                        + escape(e.detectorName()));
                w.newLine();
            }
        }
    }

    private static String escape(String s) {
        if (s.contains(",") || s.contains("\"")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }
}