package stream.ensemble;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class SubspaceManager {

    public record AdaptationResult(List<Integer> modelsChanged,
                                   List<Integer> modelsToReset,
                                   List<Set<Integer>> oldSubspaces,
                                   List<Set<Integer>> newSubspaces) {}

    private final int ensembleSize;
    private final int subspaceSize;
    private final Random rng;
    private List<Set<Integer>> subspaces;
    private int totalFeatures;
    private int classIndex;

    public SubspaceManager(int ensembleSize, int subspaceSize, long seed) {
        if (ensembleSize <= 0) throw new IllegalArgumentException("ensembleSize > 0");
        if (subspaceSize <= 0) throw new IllegalArgumentException("subspaceSize > 0");
        this.ensembleSize = ensembleSize;
        this.subspaceSize = subspaceSize;
        this.rng = new Random(seed);
        this.subspaces = new ArrayList<>();
    }

    public int ensembleSize() {
        return ensembleSize;
    }

    public int subspaceSize() {
        return subspaceSize;
    }

    public Set<Integer> subspace(int modelIndex) {
        return subspaces.get(modelIndex);
    }

    public List<Set<Integer>> subspaces() {
        return subspaces;
    }

    public void initRandom(int totalFeatures, int classIndex) {
        if (totalFeatures <= 1) {
            throw new IllegalArgumentException("totalFeatures must be > 1");
        }
        this.totalFeatures = totalFeatures;
        this.classIndex = classIndex;

        List<Integer> pool = featurePool();
        int k = Math.min(subspaceSize, pool.size());

        this.subspaces = new ArrayList<>(ensembleSize);
        for (int m = 0; m < ensembleSize; m++) {
            Collections.shuffle(pool, rng);
            Set<Integer> sub = new LinkedHashSet<>(pool.subList(0, k));
            subspaces.add(sub);
        }
    }

    public AdaptationResult adaptSubspaces(Set<Integer> weakFeatures,
                                           Set<Integer> strongFeatures,
                                           int minOverlap,
                                           double resetRatio) {
        List<Integer> modelsChanged = new ArrayList<>();
        List<Integer> modelsToReset = new ArrayList<>();
        List<Set<Integer>> oldList = new ArrayList<>();
        List<Set<Integer>> newList = new ArrayList<>();

        Set<Integer> strongPool = new LinkedHashSet<>(strongFeatures);
        strongPool.remove(classIndex);

        for (int m = 0; m < ensembleSize; m++) {
            Set<Integer> sub = subspaces.get(m);
            Set<Integer> overlap = intersection(sub, weakFeatures);
            if (overlap.size() < minOverlap) continue;

            Set<Integer> oldSub = new LinkedHashSet<>(sub);
            sub.removeAll(overlap);
            List<Integer> replacements = pickReplacements(strongPool, overlap.size(), sub, weakFeatures);
            sub.addAll(replacements);
            if (sub.size() < subspaceSize) {
                int needed = subspaceSize - sub.size();
                List<Integer> filler = pickReplacements(featurePoolSet(), needed, sub, weakFeatures);
                sub.addAll(filler);
            }
            if (sub.size() > subspaceSize) {
                List<Integer> shrink = new ArrayList<>(sub);
                Collections.shuffle(shrink, rng);
                sub.clear();
                sub.addAll(shrink.subList(0, subspaceSize));
            }

            modelsChanged.add(m);
            oldList.add(oldSub);
            newList.add(new LinkedHashSet<>(sub));

            double overlapRatio = (double) overlap.size() / Math.max(1, oldSub.size());
            if (overlapRatio >= resetRatio) {
                modelsToReset.add(m);
            }
        }
        return new AdaptationResult(modelsChanged, modelsToReset, oldList, newList);
    }

    public AdaptationResult adaptSubspaces(Set<Integer> weakFeatures,
                                           Set<Integer> strongFeatures) {
        return adaptSubspaces(weakFeatures, strongFeatures, 1, 0.0);
    }

    private List<Integer> pickReplacements(Set<Integer> candidates, int count,
                                           Set<Integer> exclude,
                                           Set<Integer> forbidden) {
        List<Integer> available = new ArrayList<>();
        for (int f : candidates) {
            if (!exclude.contains(f) && !forbidden.contains(f) && f != classIndex) {
                available.add(f);
            }
        }
        Collections.shuffle(available, rng);
        if (available.size() >= count) {
            return new ArrayList<>(available.subList(0, count));
        }
        List<Integer> picked = new ArrayList<>(available);
        Set<Integer> taken = new HashSet<>(exclude);
        taken.addAll(picked);
        taken.addAll(forbidden);
        List<Integer> fallback = new ArrayList<>();
        for (int f : featurePool()) {
            if (!taken.contains(f)) fallback.add(f);
        }
        Collections.shuffle(fallback, rng);
        for (int i = 0; i < fallback.size() && picked.size() < count; i++) {
            picked.add(fallback.get(i));
        }
        return picked;
    }

    private List<Integer> featurePool() {
        List<Integer> pool = new ArrayList<>(totalFeatures - 1);
        for (int i = 0; i < totalFeatures; i++) {
            if (i != classIndex) pool.add(i);
        }
        return pool;
    }

    private Set<Integer> featurePoolSet() {
        return new LinkedHashSet<>(featurePool());
    }

    private static Set<Integer> intersection(Set<Integer> a, Set<Integer> b) {
        Set<Integer> r = new LinkedHashSet<>();
        for (int x : a) if (b.contains(x)) r.add(x);
        return r;
    }
}
