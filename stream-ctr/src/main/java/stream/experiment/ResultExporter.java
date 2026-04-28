package stream.experiment;

import stream.drift.DriftEvent;
import stream.evaluation.EvalRecord;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class ResultExporter {

    public void exportLong(List<ExperimentResult> results, Path outPath) throws IOException {
        ensureDir(outPath);
        Set<String> metricNames = collectMetricNames(results);
        try (BufferedWriter w = Files.newBufferedWriter(outPath)) {
            w.write("dataset,model,selector,instance");
            for (String m : metricNames) {
                w.write(",");
                w.write(escape(m));
            }
            w.write(",drift_detected");
            w.newLine();

            for (ExperimentResult r : results) {
                Set<Long> driftIdx = new LinkedHashSet<>();
                for (DriftEvent e : r.driftEvents()) driftIdx.add(e.instanceIndex());

                long previousBucket = 0;
                for (EvalRecord rec : r.history()) {
                    long bucket = rec.instanceIndex();
                    long bucketStart = previousBucket;
                    long bucketEnd = bucket;
                    boolean drift = driftIdx.stream()
                            .anyMatch(d -> d >= bucketStart && d < bucketEnd);
                    w.write(esc(r.datasetName()) + "," + esc(r.modelName()) + ","
                            + esc(r.selectorName()) + "," + bucket);
                    for (String m : metricNames) {
                        Double v = rec.values().get(m);
                        w.write(",");
                        w.write(v == null ? "" : Double.toString(v));
                    }
                    w.write(",");
                    w.write(drift ? "true" : "false");
                    w.newLine();
                    previousBucket = bucket;
                }
            }
        }
    }

    public void exportSummary(List<ExperimentResult> results, Path outPath) throws IOException {
        ensureDir(outPath);
        Set<String> metricNames = collectMetricNames(results);
        try (BufferedWriter w = Files.newBufferedWriter(outPath)) {
            w.write("dataset,model,selector,instances,elapsed_ms,drifts");
            for (String m : metricNames) {
                w.write(",final_" + escape(m));
            }
            w.newLine();
            for (ExperimentResult r : results) {
                w.write(esc(r.datasetName()) + "," + esc(r.modelName()) + ","
                        + esc(r.selectorName()) + ","
                        + r.instances() + "," + r.elapsedMs() + ","
                        + r.driftEvents().size());
                for (String m : metricNames) {
                    Double v = r.finalMetrics().get(m);
                    w.write(",");
                    w.write(v == null ? "" : Double.toString(v));
                }
                w.newLine();
            }
        }
    }

    public void exportDriftEvents(List<ExperimentResult> results, Path outPath) throws IOException {
        ensureDir(outPath);
        try (BufferedWriter w = Files.newBufferedWriter(outPath)) {
            w.write("dataset,model,selector,instance,timestamp,windowBefore,windowAfter,detector");
            w.newLine();
            for (ExperimentResult r : results) {
                for (DriftEvent e : r.driftEvents()) {
                    w.write(esc(r.datasetName()) + "," + esc(r.modelName()) + ","
                            + esc(r.selectorName()) + ","
                            + e.instanceIndex() + "," + e.timestamp() + ","
                            + e.windowSizeBefore() + "," + e.windowSizeAfter() + ","
                            + esc(e.detectorName()));
                    w.newLine();
                }
            }
        }
    }

    private static Set<String> collectMetricNames(List<ExperimentResult> results) {
        Set<String> names = new LinkedHashSet<>();
        for (ExperimentResult r : results) {
            for (EvalRecord rec : r.history()) names.addAll(rec.values().keySet());
            names.addAll(r.finalMetrics().keySet());
        }
        return names;
    }

    private static void ensureDir(Path p) throws IOException {
        if (p.getParent() != null) Files.createDirectories(p.getParent());
    }

    private static String esc(String s) { return escape(s); }

    private static String escape(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n")) {
            return "\"" + s.replace("\"", "\"\"") + "\"";
        }
        return s;
    }
}
