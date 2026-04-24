package stream.model;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.trees.HoeffdingTree;

public class HoeffdingTreeModel implements StreamModel {

    private final String name;
    private final int gracePeriod;
    private final double splitConfidence;
    private final double tieThreshold;

    private HoeffdingTree classifier;
    private Instances header;
    private boolean initialized;

    public HoeffdingTreeModel() {
        this("HoeffdingTree", 200, 1e-7, 0.05);
    }

    public HoeffdingTreeModel(String name, int gracePeriod,
                              double splitConfidence, double tieThreshold) {
        this.name = name;
        this.gracePeriod = gracePeriod;
        this.splitConfidence = splitConfidence;
        this.tieThreshold = tieThreshold;
    }

    @Override
    public void initialize(Instances header) {
        this.header = header;
        buildClassifier();
    }

    private void buildClassifier() {
        if (header == null) {
            throw new IllegalStateException(
                    "Call initialize(header) before using the model");
        }
        classifier = new HoeffdingTree();
        classifier.gracePeriodOption.setValue(gracePeriod);
        classifier.splitConfidenceOption.setValue(splitConfidence);
        classifier.tieThresholdOption.setValue(tieThreshold);
        classifier.setModelContext(new com.yahoo.labs.samoa.instances.InstancesHeader(header));
        classifier.prepareForUse();
        initialized = true;
    }

    private void ensureReady() {
        if (!initialized) {
            throw new IllegalStateException(
                    "Model not initialized. Call initialize(header) first.");
        }
    }

    @Override
    public double predictProbability(Instance instance) {
        ensureReady();
        double[] votes = classifier.getVotesForInstance(instance);
        if (votes == null || votes.length == 0) {
            return 0.5;
        }
        double v0 = votes.length > 0 ? votes[0] : 0.0;
        double v1 = votes.length > 1 ? votes[1] : 0.0;
        double sum = v0 + v1;
        if (sum <= 0.0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
            return 0.5;
        }
        double p = v1 / sum;
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        return p;
    }

    @Override
    public void train(Instance instance) {
        ensureReady();
        classifier.trainOnInstance(instance);
    }

    @Override
    public void reset() {
        buildClassifier();
    }

    @Override
    public String getName() {
        return name;
    }
}