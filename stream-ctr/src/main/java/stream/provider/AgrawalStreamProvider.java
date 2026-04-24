package stream.provider;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.streams.generators.AgrawalGenerator;

public class AgrawalStreamProvider implements StreamProvider {

    private final int totalInstances;
    private final int driftPoint;
    private final int driftWidth;
    private final int functionBefore;
    private final int functionAfter;
    private final int seed;

    private AgrawalGenerator genBefore;
    private AgrawalGenerator genAfter;
    private long produced;

    public AgrawalStreamProvider(int totalInstances, int driftPoint,
                                 int functionBefore, int functionAfter) {
        this(totalInstances, driftPoint, 0, functionBefore, functionAfter, 1);
    }

    public AgrawalStreamProvider(int totalInstances, int driftPoint, int driftWidth,
                                 int functionBefore, int functionAfter, int seed) {
        if (driftPoint < 0 || driftPoint > totalInstances) {
            throw new IllegalArgumentException("driftPoint out of range");
        }
        if (driftWidth < 0) {
            throw new IllegalArgumentException("driftWidth must be >= 0");
        }
        this.totalInstances = totalInstances;
        this.driftPoint = driftPoint;
        this.driftWidth = driftWidth;
        this.functionBefore = functionBefore;
        this.functionAfter = functionAfter;
        this.seed = seed;
        restart();
    }

    private AgrawalGenerator buildGenerator(int function) {
        AgrawalGenerator g = new AgrawalGenerator();
        g.functionOption.setValue(function);
        g.instanceRandomSeedOption.setValue(seed);
        g.balanceClassesOption.setValue(false);
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
        Instance inst;
        if (driftWidth == 0) {
            if (produced < driftPoint) {
                inst = genBefore.nextInstance().getData();
            } else {
                inst = genAfter.nextInstance().getData();
            }
        } else {
            int start = driftPoint - driftWidth / 2;
            int end = driftPoint + driftWidth / 2;
            if (produced < start) {
                inst = genBefore.nextInstance().getData();
            } else if (produced >= end) {
                inst = genAfter.nextInstance().getData();
            } else {
                double progress = (produced - start) / (double) driftWidth;
                boolean useAfter = Math.random() < progress;
                inst = useAfter
                        ? genAfter.nextInstance().getData()
                        : genBefore.nextInstance().getData();
            }
        }
        produced++;
        return inst;
    }

    @Override
    public void restart() {
        this.genBefore = buildGenerator(functionBefore);
        this.genAfter = buildGenerator(functionAfter);
        this.produced = 0;
    }

    public long produced() {
        return produced;
    }

    public int driftPoint() {
        return driftPoint;
    }
}