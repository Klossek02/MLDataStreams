package stream.model;

public final class ProbabilityUtils {

    private ProbabilityUtils() {}

    public static double normalizeBinary(double[] votes) {
        if (votes == null || votes.length == 0) {
            return 0.5;
        }
        double v0 = votes.length > 0 ? votes[0] : 0.0;
        double v1 = votes.length > 1 ? votes[1] : 0.0;
        if (Double.isNaN(v0) || Double.isNaN(v1)
                || Double.isInfinite(v0) || Double.isInfinite(v1)) {
            return 0.5;
        }
        double sum = v0 + v1;
        if (sum <= 0.0) {
            return 0.5;
        }
        double p = v1 / sum;
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        return p;
    }
}