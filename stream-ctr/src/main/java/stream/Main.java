package stream;

import stream.experiment.RunDriftAwareExperiments;
import stream.experiment.RunExperiments;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        long t0 = System.currentTimeMillis();

        System.out.println("############################################");
        System.out.println("### PHASE 1/2: BASELINE EXPERIMENTS      ###");
        System.out.println("############################################");
        RunExperiments.main(args);

        System.out.println();
        System.out.println("############################################");
        System.out.println("### PHASE 2/2: DRIFT-AWARE EXPERIMENTS   ###");
        System.out.println("############################################");
        RunDriftAwareExperiments.main(args);

        long elapsed = System.currentTimeMillis() - t0;
        System.out.println();
        System.out.printf("=== ALL DONE in %d s ===%n", elapsed / 1000);
        System.out.println("CSV files in results/:");
        System.out.println("  results_long.csv      results_summary.csv      results_drifts.csv");
        System.out.println("  driftaware_long.csv   driftaware_summary.csv   driftaware_drifts.csv");
    }
}