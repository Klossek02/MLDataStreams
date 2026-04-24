package stream.model;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.trees.HoeffdingAdaptiveTree;

public class HatModel implements StreamModel {

    private final String name;
    private final int gracePeriod;
    private final double splitConfidence;
    private final double tieThreshold;

    private HoeffdingAdaptiveTree classifier;
    private Instances header;
    private boolean initialized;

    public HatModel() {
        this("HAT", 200, 1e-7, 0.05);
    }

    public HatModel(String name, int gracePeriod,
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
        classifier = new HoeffdingAdaptiveTree();
        classifier.gracePeriodOption.setValue(gracePeriod);
        classifier.splitConfidenceOption.setValue(splitConfidence);
        classifier.tieThresholdOption.setValue(tieThreshold);
        classifier.setModelContext(new InstancesHeader(header));
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
        return ProbabilityUtils.normalizeBinary(votes);
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