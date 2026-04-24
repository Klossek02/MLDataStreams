package stream.evaluation;

public class LogLossMetric implements Metric {

    private static final double EPS = 1e-15;

    private double sumLoss;
    private long count;

    @Override
    public void update(double predictedProb, int trueLabel) {
        double p = Math.max(EPS, Math.min(1.0 - EPS, predictedProb));
        double loss = -(trueLabel * Math.log(p) + (1 - trueLabel) * Math.log(1 - p));
        sumLoss += loss;
        count++;
    }

    @Override
    public double getValue() {
        return count == 0 ? 0.0 : sumLoss / count;
    }

    @Override
    public String getName() {
        return "LogLoss";
    }

    @Override
    public void reset() {
        sumLoss = 0.0;
        count = 0;
    }
}