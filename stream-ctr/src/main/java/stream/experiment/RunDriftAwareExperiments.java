package stream.experiment;

import stream.model.HatModel;
import stream.model.HoeffdingTreeModel;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class RunDriftAwareExperiments {

    public static void main(String[] args) throws IOException {
        String avazuPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_extended.arff";
        String criteoPath = (args.length > 1)
                ? args[1]
                : "/home/kubog/MLDataStreams/criteo/data/criteo_extended.arff";
        Path outDir = Path.of("results");

        List<DriftAwareExperimentBuilder.Spec> specs =
                DriftAwareExperimentBuilder.defaultSpecs(avazuPath, criteoPath);

        List<ExperimentResult> results = new ArrayList<>();
        int total = specs.size() * 3;
        int idx = 0;
        long globalStart = System.currentTimeMillis();

        for (var spec : specs) {
            idx++;
            banner(idx, total, spec.datasetName(), "DriftAwareSelector + HT", globalStart);
            results.add(safeRun(() -> DriftAwareExperimentBuilder
                    .runDriftAwareSelectorWithModel(spec, "HT", HoeffdingTreeModel::new)));

            idx++;
            banner(idx, total, spec.datasetName(), "DriftAwareSelector + HAT", globalStart);
            results.add(safeRun(() -> DriftAwareExperimentBuilder
                    .runDriftAwareSelectorWithModel(spec, "HAT", HatModel::new)));

            idx++;
            banner(idx, total, spec.datasetName(), "DriftAwareSRP", globalStart);
            results.add(safeRun(() -> DriftAwareExperimentBuilder.runDriftAwareSrp(spec)));
        }

        results.removeIf(r -> r == null);

        ResultExporter ex = new ResultExporter();
        Path longCsv = outDir.resolve("driftaware_long.csv");
        Path summaryCsv = outDir.resolve("driftaware_summary.csv");
        Path driftsCsv = outDir.resolve("driftaware_drifts.csv");
        ex.exportLong(results, longCsv);
        ex.exportSummary(results, summaryCsv);
        ex.exportDriftEvents(results, driftsCsv);

        System.out.println();
        System.out.println("Wrote:");
        System.out.println("  " + longCsv.toAbsolutePath());
        System.out.println("  " + summaryCsv.toAbsolutePath());
        System.out.println("  " + driftsCsv.toAbsolutePath());

        System.out.println();
        System.out.println("=== Summary table (drift-aware) ===");
        System.out.printf("%-18s %-6s %-22s %12s %10s %8s%n",
                "dataset", "model", "selector", "logloss", "auc", "drifts");
        for (var r : results) {
            System.out.printf("%-18s %-6s %-22s %12.4f %10.4f %8d%n",
                    r.datasetName(), r.modelName(), r.selectorName(),
                    r.finalMetrics().getOrDefault("LogLoss", Double.NaN),
                    r.finalMetrics().getOrDefault("AUC", Double.NaN),
                    r.driftEvents().size());
        }

        long totalSec = (System.currentTimeMillis() - globalStart) / 1000;
        System.out.printf("=== Drift-aware suite finished in %d s ===%n", totalSec);
    }

    private static void banner(int idx, int total, String dataset, String what, long globalStart) {
        long elapsed = (System.currentTimeMillis() - globalStart) / 1000;
        System.out.printf("%n[%d/%d] (suite elapsed=%ds) %s + %s%n",
                idx, total, elapsed, dataset, what);
    }

    private static ExperimentResult safeRun(java.util.function.Supplier<ExperimentResult> fn) {
        try {
            ExperimentResult r = fn.get();
            System.out.printf("  done n=%d ms=%d %s%n",
                    r.instances(), r.elapsedMs(), r.finalMetrics());
            return r;
        } catch (Exception e) {
            System.err.println("  FAILED: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}