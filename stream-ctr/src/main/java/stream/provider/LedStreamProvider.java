package stream.provider;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.streams.generators.LEDGeneratorDrift;

public class LedStreamProvider implements StreamProvider {

    private final int totalInstances;
    private final int driftPoint;
    private final int attrsDriftBefore;
    private final int attrsDriftAfter;
    private final int seed;

    private LEDGeneratorDrift genBefore;
    private LEDGeneratorDrift genAfter;
    private long produced;

    public LedStreamProvider(int totalInstances, int driftPoint,
                             int attrsDriftBefore, int attrsDriftAfter,
                             int seed) {
        if (driftPoint < 0 || driftPoint > totalInstances) {
            throw new IllegalArgumentException("driftPoint out of range");
        }
        this.totalInstances = totalInstances;
        this.driftPoint = driftPoint;
        this.attrsDriftBefore = attrsDriftBefore;
        this.attrsDriftAfter = attrsDriftAfter;
        this.seed = seed;
        restart();
    }

    private LEDGeneratorDrift buildGenerator(int numAttrsDrift) {
        LEDGeneratorDrift g = new LEDGeneratorDrift();
        g.numberAttributesDriftOption.setValue(numAttrsDrift);
        g.instanceRandomSeedOption.setValue(seed);
        g.prepareForUse();
        return g;
    }

    @Override
    public Instances getHeader() {
        return genBefore.getHeader();
    }

    @Override
    public boolean hasNext() {
        return produced < totalInstances
                && genBefore.hasMoreInstances()
                && genAfter.hasMoreInstances();
    }

    @Override
    public Instance next() {
        if (!hasNext()) {
            throw new IllegalStateException("No more instances");
        }
        Instance inst = produced < driftPoint
                ? genBefore.nextInstance().getData()
                : genAfter.nextInstance().getData();
        produced++;
        return inst;
    }

    @Override
    public void restart() {
        this.genBefore = buildGenerator(attrsDriftBefore);
        this.genAfter = buildGenerator(attrsDriftAfter);
        this.produced = 0;
    }
}
