package stream.ensemble;

import java.util.Arrays;
import java.util.List;

public class SmokeTestWeightManager {

    public static void main(String[] args) {
        System.out.println("=== Test 1: initial weights = normalWeight ===");
        WeightManager wm = new WeightManager(5);
        for (int i = 0; i < 5; i++) {
            assertEq(wm.weight(i), 1.0, "initial weight " + i);
        }
        System.out.println("OK: " + Arrays.toString(wm.weightsSnapshot()));

        System.out.println();
        System.out.println("=== Test 2: onModelsReset lowers selected weights ===");
        wm.onModelsReset(List.of(1, 3));
        assertEq(wm.weight(0), 1.0, "m0 untouched");
        assertEq(wm.weight(1), 0.3, "m1 reset");
        assertEq(wm.weight(2), 1.0, "m2 untouched");
        assertEq(wm.weight(3), 0.3, "m3 reset");
        assertEq(wm.weight(4), 1.0, "m4 untouched");
        System.out.println("OK: " + Arrays.toString(wm.weightsSnapshot()));

        System.out.println();
        System.out.println("=== Test 3: decay raises reset weights but caps at normalWeight ===");
        WeightManager wm2 = new WeightManager(3, 0.2, 1.0, 0.1);
        wm2.onModelsReset(List.of(0, 1));
        for (int i = 0; i < 5; i++) wm2.decay();
        assertEq(wm2.weight(0), 0.7, "m0 after 5 decays (0.2+5*0.1)");
        assertEq(wm2.weight(1), 0.7, "m1 after 5 decays");
        assertEq(wm2.weight(2), 1.0, "m2 untouched and not above 1.0");
        for (int i = 0; i < 100; i++) wm2.decay();
        assertEq(wm2.weight(0), 1.0, "m0 capped at 1.0");
        assertEq(wm2.weight(1), 1.0, "m1 capped at 1.0");
        System.out.println("OK: cap works, " + Arrays.toString(wm2.weightsSnapshot()));

        System.out.println();
        System.out.println("=== Test 4: weightedPrediction with equal weights ===");
        WeightManager wm3 = new WeightManager(4);
        double p = wm3.weightedPrediction(new double[]{0.0, 0.0, 1.0, 1.0});
        assertEq(p, 0.5, "equal weights, mean = 0.5");
        System.out.println("OK: equal-weight average = " + p);

        System.out.println();
        System.out.println("=== Test 5: weightedPrediction respects reset weights ===");
        WeightManager wm4 = new WeightManager(2, 0.0, 1.0, 0.001);
        wm4.onModelsReset(List.of(1));
        double p2 = wm4.weightedPrediction(new double[]{0.2, 0.9});
        assertEq(p2, 0.2, "model 1 has weight 0 -> only model 0 counts");
        System.out.println("OK: weighted = " + p2);

        System.out.println();
        System.out.println("=== Test 6: weightedPrediction safe when all weights = 0 ===");
        WeightManager wm5 = new WeightManager(2, 0.0, 1.0, 0.001);
        wm5.onModelsReset(List.of(0, 1));
        double p3 = wm5.weightedPrediction(new double[]{0.7, 0.4});
        assertEq(p3, 0.5, "all weights 0 -> fallback 0.5");
        System.out.println("OK: fallback = " + p3);

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void assertEq(double actual, double expected, String msg) {
        if (Math.abs(actual - expected) > 1e-9) {
            throw new AssertionError("FAIL " + msg + ": expected " + expected + ", got " + actual);
        }
    }
}