package stream.model;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.meta.StreamingRandomPatches;

public class SrpModel implements StreamModel {

    private final String name;
    private final int ensembleSize;
    private final int subspaceSize;
    private final int seed;

    private StreamingRandomPatches classifier;
    private Instances header;
    private boolean initialized;

    public SrpModel() {
        this("SRP", 10, 60, 1);
    }

    public SrpModel(String name, int ensembleSize, int subspaceSize, int seed) {
        this.name = name;
        this.ensembleSize = ensembleSize;
        this.subspaceSize = subspaceSize;
        this.seed = seed;
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
        classifier = new StreamingRandomPatches();
        classifier.ensembleSizeOption.setValue(ensembleSize);
        classifier.subspaceSizeOption.setValue(subspaceSize);
        trySetRandomSeed(classifier, seed);
        classifier.setModelContext(new InstancesHeader(header));
        classifier.prepareForUse();
        initialized = true;
    }

    private static void trySetRandomSeed(Object classifier, int seed) {
        try {
            java.lang.reflect.Field f = moa.classifiers.AbstractClassifier.class
                    .getDeclaredField("randomSeedOption");
            f.setAccessible(true);
            Object opt = f.get(classifier);
            if (opt instanceof com.github.javacliparser.IntOption io) {
                io.setValue(seed);
            }
        } catch (ReflectiveOperationException ignored) {
        }
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