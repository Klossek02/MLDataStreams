package stream.config;

public record ExperimentConfig(
        String datasetPath,
        String datasetName,
        int windowSize,
        int topK,
        double adwinDelta,
        int srpEnsembleSize,
        int srpSubspaceSize,
        double changeThreshold,
        String outputDir
) {
    public static ExperimentConfig avazuDefault() {
        return new ExperimentConfig(
                ProjectPaths.avazuHashedArff(),
                "avazu",
                1000,
                20,
                0.002,
                10,
                10,
                0.01,
                "results/"
        );
    }

    public static ExperimentConfig criteoDefault() {
        return new ExperimentConfig(
                ProjectPaths.criteoHashedArff(),
                "criteo",
                1000,
                20,
                0.002,
                10,
                10,
                0.01,
                "results/"
        );
    }

    public static ExperimentConfig agrawalDefault() {
        return new ExperimentConfig(
                null,
                "agrawal",
                1000,
                5,
                0.002,
                10,
                3,
                0.01,
                "results/"
        );
    }
}
