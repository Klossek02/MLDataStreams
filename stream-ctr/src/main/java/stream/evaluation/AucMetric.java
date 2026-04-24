package stream.evaluation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class AucMetric implements Metric {

    private final int capacity;
    private final double[] probs;
    private final byte[] labels;
    private int size;
    private int writeIndex;
    private boolean filled;

    public AucMetric() {
        this(10_000);
    }

    public AucMetric(int windowCapacity) {
        if (windowCapacity <= 0) {
            throw new IllegalArgumentException("windowCapacity must be > 0");
        }
        this.capacity = windowCapacity;
        this.probs = new double[capacity];
        this.labels = new byte[capacity];
    }

    @Override
    public void update(double predictedProb, int trueLabel) {
        probs[writeIndex] = predictedProb;
        labels[writeIndex] = (byte) trueLabel;
        writeIndex = (writeIndex + 1) % capacity;
        if (!filled) {
            size++;
            if (size == capacity) filled = true;
        }
    }

    @Override
    public double getValue() {
        int n = filled ? capacity : size;
        if (n < 2) return 0.5;

        long pos = 0, neg = 0;
        for (int i = 0; i < n; i++) {
            if (labels[i] == 1) pos++; else neg++;
        }
        if (pos == 0 || neg == 0) return 0.5;

        List<int[]> idx = new ArrayList<>(n);
        for (int i = 0; i < n; i++) idx.add(new int[]{i});
        idx.sort(Comparator.comparingDouble(a -> probs[a[0]]));

        double rankSum = 0.0;
        int i = 0;
        long rank = 1;
        while (i < n) {
            int j = i;
            double v = probs[idx.get(i)[0]];
            while (j < n && probs[idx.get(j)[0]] == v) j++;
            double avgRank = (rank + (rank + (j - i) - 1)) / 2.0;
            for (int k = i; k < j; k++) {
                if (labels[idx.get(k)[0]] == 1) {
                    rankSum += avgRank;
                }
            }
            rank += (j - i);
            i = j;
        }

        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    @Override
    public String getName() {
        return "AUC";
    }

    @Override
    public void reset() {
        size = 0;
        writeIndex = 0;
        filled = false;
    }
}