package stream.evaluation;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.Supplier;

public class WindowedMetric implements Metric {

    private final int windowSize;
    private final Supplier<Metric> factory;
    private final Metric inner;
    private final Deque<double[]> buffer;

    public WindowedMetric(int windowSize, Supplier<Metric> factory) {
        if (windowSize <= 0) {
            throw new IllegalArgumentException("windowSize must be > 0");
        }
        this.windowSize = windowSize;
        this.factory = factory;
        this.inner = factory.get();
        this.buffer = new ArrayDeque<>(windowSize);
    }

    @Override
    public void update(double predictedProb, int trueLabel) {
        buffer.addLast(new double[]{predictedProb, trueLabel});
        if (buffer.size() > windowSize) {
            buffer.removeFirst();
        }
    }

    @Override
    public double getValue() {
        inner.reset();
        for (double[] pair : buffer) {
            inner.update(pair[0], (int) pair[1]);
        }
        return inner.getValue();
    }

    @Override
    public String getName() {
        return "Windowed[" + windowSize + "]-" + factory.get().getName();
    }

    @Override
    public void reset() {
        buffer.clear();
        inner.reset();
    }

    public int currentSize() {
        return buffer.size();
    }
}