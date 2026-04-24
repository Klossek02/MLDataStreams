package stream.ensemble;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class SmokeTestSubspaceManager {

    public static void main(String[] args) {
        System.out.println("=== Test 1: initRandom basic invariants ===");
        SubspaceManager sm = new SubspaceManager(10, 5, 42);
        sm.initRandom(100, 99);
        for (int i = 0; i < sm.ensembleSize(); i++) {
            Set<Integer> sub = sm.subspace(i);
            assertCond(sub.size() == 5, "subspace size = 5");
            assertCond(!sub.contains(99), "class index not in subspace");
            for (int f : sub) {
                assertCond(f >= 0 && f < 100, "feature in range");
            }
        }
        System.out.println("OK: 10 subspaces of size 5 from 100 features");

        System.out.println();
        System.out.println("=== Test 2: adaptSubspaces removes weak, adds strong ===");
        SubspaceManager sm2 = new SubspaceManager(5, 4, 1);
        sm2.initRandom(20, 19);
        System.out.println("Before:");
        for (int i = 0; i < sm2.ensembleSize(); i++) {
            System.out.println("  m" + i + " = " + sm2.subspace(i));
        }
        Set<Integer> weak = new LinkedHashSet<>();
        for (int f : sm2.subspace(0)) { weak.add(f); break; }
        Set<Integer> strong = Set.of(15, 16, 17, 18);

        SubspaceManager.AdaptationResult res = sm2.adaptSubspaces(weak, strong);
        System.out.println("Weak=" + weak + " Strong=" + strong);
        System.out.println("Models reset: " + res.modelsToReset());
        System.out.println("After:");
        for (int i = 0; i < sm2.ensembleSize(); i++) {
            System.out.println("  m" + i + " = " + sm2.subspace(i));
        }
        for (int m : res.modelsToReset()) {
            assertCond(java.util.Collections.disjoint(sm2.subspace(m), weak),
                    "weak features removed from m" + m);
            assertCond(sm2.subspace(m).size() == 4,
                    "subspace size preserved for m" + m);
        }

        System.out.println();
        System.out.println("=== Test 3: adaptSubspaces no overlap -> no resets ===");
        SubspaceManager sm3 = new SubspaceManager(5, 4, 7);
        sm3.initRandom(20, 19);
        Set<Integer> weakNone = Set.of(99);
        SubspaceManager.AdaptationResult res3 = sm3.adaptSubspaces(weakNone, Set.of());
        assertCond(res3.modelsToReset().isEmpty(),
                "no resets when no overlap");
        System.out.println("OK: no resets triggered");

        System.out.println();
        System.out.println("=== Test 4: adaptSubspaces falls back when strong pool too small ===");
        SubspaceManager sm4 = new SubspaceManager(3, 5, 11);
        sm4.initRandom(15, 14);
        Set<Integer> weak4 = new LinkedHashSet<>(sm4.subspace(0));
        Set<Integer> strong4 = Set.of();
        SubspaceManager.AdaptationResult res4 = sm4.adaptSubspaces(weak4, strong4);
        for (int m : res4.modelsToReset()) {
            assertCond(sm4.subspace(m).size() == 5,
                    "subspace refilled to size 5 even without strong pool, m=" + m);
            assertCond(java.util.Collections.disjoint(sm4.subspace(m), weak4),
                    "weak fully replaced m=" + m);
        }
        System.out.println("OK: fallback to random fillers works");

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void assertCond(boolean cond, String msg) {
        if (!cond) throw new AssertionError("FAIL: " + msg);
    }
}