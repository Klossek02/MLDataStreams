package stream.experiment;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class RunExperiments {

    public static void main(String[] args) throws IOException {
        String avazuPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_extended.arff";
        String criteoPath = (args.length > 1)
                ? args[1]
                : "/home/kubog/MLDataStreams/criteo/data/criteo_extended.arff";
        Path outDir = Path.of("results");

        List<Experiment> matrix = ExperimentMatrix.build(avazuPath, criteoPath);
        System.out.println("Total experiments: " + matrix.size());

        ExperimentRunner runner = new ExperimentRunner();
        List<ExperimentResult> results = runner.runAll(matrix);

        ResultExporter ex = new ResultExporter();
        Path longCsv = outDir.resolve("results_long.csv");
        Path summaryCsv = outDir.resolve("results_summary.csv");
        Path driftsCsv = outDir.resolve("results_drifts.csv");

        ex.exportLong(results, longCsv);
        ex.exportSummary(results, summaryCsv);
        ex.exportDriftEvents(results, driftsCsv);

        System.out.println();
        System.out.println("Wrote:");
        System.out.println("  " + longCsv.toAbsolutePath());
        System.out.println("  " + summaryCsv.toAbsolutePath());
        System.out.println("  " + driftsCsv.toAbsolutePath());

        System.out.println();
        System.out.println("=== Summary table ===");
        System.out.printf("%-18s %-6s %-16s %12s %10s %8s%n",
                "dataset", "model", "selector", "logloss", "auc", "drifts");
        for (ExperimentResult r : results) {
            System.out.printf("%-18s %-6s %-16s %12.4f %10.4f %8d%n",
                    r.datasetName(), r.modelName(), r.selectorName(),
                    r.finalMetrics().getOrDefault("LogLoss", Double.NaN),
                    r.finalMetrics().getOrDefault("AUC", Double.NaN),
                    r.driftEvents().size());
        }
    }
}