package stream.evaluation;

public interface Metric {

    void update(double predictedProb, int trueLabel);

    double getValue();

    String getName();

    void reset();
}