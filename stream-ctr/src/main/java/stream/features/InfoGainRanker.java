package stream.features;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InfoGainRanker {

    private final int numericBins;

    public InfoGainRanker() {
        this(10);
    }

    public InfoGainRanker(int numericBins) {
        if (numericBins < 2) throw new IllegalArgumentException("numericBins >= 2");
        this.numericBins = numericBins;
    }

    public Map<Integer, Double> rank(Instances buffer, List<Instance> samples) {
        int classIdx = buffer.classIndex();
        int numClasses = buffer.numClasses();
        int n = samples.size();

        int[] classCounts = new int[numClasses];
        for (Instance ins : samples) classCounts[(int) ins.classValue()]++;
        double hY = entropy(classCounts, n);

        Map<Integer, Double> ig = new HashMap<>();
        for (int a = 0; a < buffer.numAttributes(); a++) {
            if (a == classIdx) continue;
            Attribute attr = buffer.attribute(a);
            double hYgivenX;
            if (attr.isNominal()) {
                hYgivenX = condEntropyNominal(samples, a, attr.numValues(), numClasses, n);
            } else {
                hYgivenX = condEntropyNumeric(samples, a, numClasses, n);
            }
            ig.put(a, Math.max(0.0, hY - hYgivenX));
        }
        return ig;
    }

    private double condEntropyNominal(List<Instance> samples, int a,
                                      int numValues, int numClasses, int n) {
        int[][] joint = new int[numValues][numClasses];
        int[] valCount = new int[numValues];
        int valid = 0;
        for (Instance ins : samples) {
            if (ins.isMissing(a)) continue;
            int v = (int) ins.value(a);
            if (v < 0 || v >= numValues) continue;
            int c = (int) ins.classValue();
            joint[v][c]++;
            valCount[v]++;
            valid++;
        }
        if (valid == 0) return 0.0;
        double sum = 0.0;
        for (int v = 0; v < numValues; v++) {
            if (valCount[v] == 0) continue;
            sum += (valCount[v] / (double) valid) * entropy(joint[v], valCount[v]);
        }
        return sum;
    }

    private double condEntropyNumeric(List<Instance> samples, int a,
                                      int numClasses, int n) {
        double[] vals = new double[n];
        int valid = 0;
        for (Instance ins : samples) {
            if (ins.isMissing(a)) continue;
            vals[valid++] = ins.value(a);
        }
        if (valid == 0) return 0.0;
        double[] trimmed = Arrays.copyOf(vals, valid);
        Arrays.sort(trimmed);

        int bins = Math.min(numericBins, valid);
        double[] thresholds = new double[bins - 1];
        for (int i = 1; i < bins; i++) {
            int idx = (int) Math.floor(i * (valid / (double) bins));
            if (idx >= valid) idx = valid - 1;
            thresholds[i - 1] = trimmed[idx];
        }

        int[][] joint = new int[bins][numClasses];
        int[] binCount = new int[bins];
        for (Instance ins : samples) {
            if (ins.isMissing(a)) continue;
            double v = ins.value(a);
            int b = binIndex(v, thresholds);
            joint[b][(int) ins.classValue()]++;
            binCount[b]++;
        }
        double sum = 0.0;
        for (int b = 0; b < bins; b++) {
            if (binCount[b] == 0) continue;
            sum += (binCount[b] / (double) valid) * entropy(joint[b], binCount[b]);
        }
        return sum;
    }

    private static int binIndex(double v, double[] thresholds) {
        int idx = Arrays.binarySearch(thresholds, v);
        if (idx < 0) idx = -idx - 1;
        if (idx >= thresholds.length) idx = thresholds.length;
        return idx;
    }

    private static double entropy(int[] counts, int total) {
        if (total <= 0) return 0.0;
        double h = 0.0;
        double inv = 1.0 / total;
        double ln2 = Math.log(2);
        for (int c : counts) {
            if (c <= 0) continue;
            double p = c * inv;
            h -= p * Math.log(p) / ln2;
        }
        return h;
    }
}