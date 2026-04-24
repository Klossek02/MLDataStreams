package stream.ensemble;

import java.util.Arrays;
import java.util.List;

public class WeightManager {

    private final int ensembleSize;
    private final double resetWeight;
    private final double normalWeight;
    private final double decayRate;
    private final double[] weights;

    public WeightManager(int ensembleSize) {
        this(ensembleSize, 0.3, 1.0, 0.001);
    }

    public WeightManager(int ensembleSize,
                         double resetWeight,
                         double normalWeight,
                         double decayRate) {
        if (ensembleSize <= 0) throw new IllegalArgumentException("ensembleSize > 0");
        if (resetWeight < 0 || resetWeight > normalWeight) {
            throw new IllegalArgumentException(
                    "0 <= resetWeight <= normalWeight required");
        }
        if (decayRate <= 0) throw new IllegalArgumentException("decayRate > 0");
        this.ensembleSize = ensembleSize;
        this.resetWeight = resetWeight;
        this.normalWeight = normalWeight;
        this.decayRate = decayRate;
        this.weights = new double[ensembleSize];
        Arrays.fill(weights, normalWeight);
    }

    public int size() {
        return ensembleSize;
    }

    public double weight(int i) {
        return weights[i];
    }

    public double[] weightsSnapshot() {
        return weights.clone();
    }

    public void resetAll() {
        Arrays.fill(weights, normalWeight);
    }

    public void onModelsReset(List<Integer> resetIndices) {
        for (int idx : resetIndices) {
            if (idx < 0 || idx >= ensembleSize) {
                throw new IndexOutOfBoundsException("model index " + idx);
            }
            weights[idx] = resetWeight;
        }
    }

    public void decay() {
        for (int i = 0; i < ensembleSize; i++) {
            if (weights[i] < normalWeight) {
                weights[i] = Math.min(normalWeight, weights[i] + decayRate);
            }
        }
    }

    public double weightedPrediction(double[] probs) {
        if (probs == null || probs.length != ensembleSize) {
            throw new IllegalArgumentException(
                    "probs length must equal ensembleSize");
        }
        double sumWP = 0.0;
        double sumW = 0.0;
        for (int i = 0; i < ensembleSize; i++) {
            double w = weights[i];
            sumWP += w * probs[i];
            sumW += w;
        }
        if (sumW <= 0.0) return 0.5;
        double p = sumWP / sumW;
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        return p;
    }

    public double resetWeight() {
        return resetWeight;
    }

    public double normalWeight() {
        return normalWeight;
    }

    public double decayRate() {
        return decayRate;
    }
}