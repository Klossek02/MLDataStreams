package stream.provider;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.core.Example;
import moa.streams.ConceptDriftStream;
import moa.streams.generators.RandomRBFGenerator;

public class RandomRbfStreamProvider implements StreamProvider {

    private final int totalInstances;
    private final int driftPoint;
    private final int numAtts;
    private final int numClasses;
    private final int numCentroids;
    private final int seedBefore;
    private final int seedAfter;

    private ConceptDriftStream stream;
    private long produced;

    public RandomRbfStreamProvider(int totalInstances, int driftPoint,
                                   int numAtts, int numClasses, int numCentroids,
                                   int seedBefore, int seedAfter) {
        if (driftPoint < 0 || driftPoint > totalInstances) {
            throw new IllegalArgumentException("driftPoint out of range");
        }
        this.totalInstances = totalInstances;
        this.driftPoint = driftPoint;
        this.numAtts = numAtts;
        this.numClasses = numClasses;
        this.numCentroids = numCentroids;
        this.seedBefore = seedBefore;
        this.seedAfter = seedAfter;
        restart();
    }

    private RandomRBFGenerator buildGenerator(int seed) {
        RandomRBFGenerator g = new RandomRBFGenerator();
        g.numAttsOption.setValue(numAtts);
        g.numClassesOption.setValue(numClasses);
        g.numCentroidsOption.setValue(numCentroids);
        g.modelRandomSeedOption.setValue(seed);
        g.instanceRandomSeedOption.setValue(seed);
        g.prepareForUse();
        return g;
    }

    private ConceptDriftStream buildStream() {
        ConceptDriftStream s = new ConceptDriftStream();
        s.streamOption.setCurrentObject(buildGenerator(seedBefore));
        s.driftstreamOption.setCurrentObject(buildGenerator(seedAfter));
        s.positionOption.setValue(driftPoint);
        s.widthOption.setValue(1);
        s.prepareForUse();
        return s;
    }

    @Override
    public Instances getHeader() {
        return stream.getHeader();
    }

    @Override
    public boolean hasNext() {
        return produced < totalInstances && stream.hasMoreInstances();
    }

    @Override
    public Instance next() {
        if (!hasNext()) {
            throw new IllegalStateException("No more instances");
        }
        Example<?> example = stream.nextInstance();
        produced++;
        return (Instance) example.getData();
    }

    @Override
    public void restart() {
        this.stream = buildStream();
        this.produced = 0;
    }
}
