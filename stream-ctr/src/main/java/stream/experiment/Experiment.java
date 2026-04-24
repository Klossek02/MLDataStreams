package stream.experiment;

import stream.drift.DriftDetector;
import stream.evaluation.Metric;
import stream.features.FeatureSelector;
import stream.model.StreamModel;
import stream.provider.StreamProvider;

import java.util.List;
import java.util.function.Supplier;

public record Experiment(
        String name,
        String datasetName,
        String modelName,
        String selectorName,
        Supplier<StreamProvider> streamFactory,
        Supplier<StreamModel> modelFactory,
        Supplier<FeatureSelector> selectorFactory,
        Supplier<DriftDetector> detectorFactory,
        List<Supplier<Metric>> metricFactories,
        long limit,
        int logInterval) {

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String name;
        private String datasetName;
        private String modelName;
        private String selectorName = "none";
        private Supplier<StreamProvider> streamFactory;
        private Supplier<StreamModel> modelFactory;
        private Supplier<FeatureSelector> selectorFactory;
        private Supplier<DriftDetector> detectorFactory;
        private List<Supplier<Metric>> metricFactories = List.of();
        private long limit = -1;
        private int logInterval = 1_000;

        public Builder name(String n) { this.name = n; return this; }
        public Builder dataset(String n, Supplier<StreamProvider> f) {
            this.datasetName = n; this.streamFactory = f; return this;
        }
        public Builder model(String n, Supplier<StreamModel> f) {
            this.modelName = n; this.modelFactory = f; return this;
        }
        public Builder selector(String n, Supplier<FeatureSelector> f) {
            this.selectorName = n; this.selectorFactory = f; return this;
        }
        public Builder detector(Supplier<DriftDetector> f) {
            this.detectorFactory = f; return this;
        }
        public Builder metrics(List<Supplier<Metric>> f) {
            this.metricFactories = f; return this;
        }
        public Builder limit(long n) { this.limit = n; return this; }
        public Builder logInterval(int n) { this.logInterval = n; return this; }

        public Experiment build() {
            if (name == null) {
                name = (datasetName + "_" + modelName + "_" + selectorName).toLowerCase();
            }
            return new Experiment(name, datasetName, modelName, selectorName,
                    streamFactory, modelFactory, selectorFactory, detectorFactory,
                    metricFactories, limit, logInterval);
        }
    }
}