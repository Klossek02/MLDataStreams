package stream.evaluation;

public class AccuracyMetric implements Metric {

    private final double threshold;
    private long correct;
    private long total;

    public AccuracyMetric() {
        this(0.5);
    }

    public AccuracyMetric(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public void update(double predictedProb, int trueLabel) {
        int yHat = (predictedProb >= threshold) ? 1 : 0;
        if (yHat == trueLabel) correct++;
        total++;
    }

    @Override
    public double getValue() {
        return total == 0 ? 0.0 : correct / (double) total;
    }

    @Override
    public String getName() {
        return "Accuracy";
    }

    @Override
    public void reset() {
        correct = 0;
        total = 0;
    }
}