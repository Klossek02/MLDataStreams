package stream.evaluation;

import java.util.LinkedHashMap;
import java.util.Map;

public record EvalRecord(long instanceIndex, Map<String, Double> values) {

    public static EvalRecord of(long idx, String[] names, double[] vals) {
        Map<String, Double> map = new LinkedHashMap<>();
        for (int i = 0; i < names.length; i++) {
            map.put(names[i], vals[i]);
        }
        return new EvalRecord(idx, map);
    }
}