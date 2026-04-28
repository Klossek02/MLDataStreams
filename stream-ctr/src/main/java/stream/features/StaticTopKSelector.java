package stream.features;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class StaticTopKSelector implements FeatureSelector {

    private final int topK;
    private final int warmupSize;
    private final InfoGainRanker ranker;

    private Instances originalHeader;
    private Instances filtered;
    private final List<Instance> warmup = new ArrayList<>();
    private int[] selectedSorted;
    private Set<Integer> selectedSet;
    private boolean ready;

    public StaticTopKSelector(int topK, int warmupSize) {
        this(topK, warmupSize, new InfoGainRanker());
    }

    public StaticTopKSelector(int topK, int warmupSize, InfoGainRanker ranker) {
        this.topK = topK;
        this.warmupSize = warmupSize;
        this.ranker = ranker;
    }

    @Override
    public void initialize(Instances header) {
        if (ready && selectedSorted != null) {
            validateSelection(header);
            this.originalHeader = header;
            this.warmup.clear();
            this.filtered = InstanceFilter.createFilteredHeader(header, selectedSorted);
            return;
        }
        this.originalHeader = header;
        this.warmup.clear();
        this.ready = false;
        this.selectedSet = new LinkedHashSet<>();
        this.filtered = header;
    }

    @Override
    public Instance filter(Instance instance) {
        if (!ready) return instance;
        return InstanceFilter.filterAttributes(instance, selectedSorted, filtered);
    }

    @Override
    public void observe(Instance instance, int trueLabel) {
        if (ready) return;
        warmup.add(instance);
        if (warmup.size() >= warmupSize) {
            computeRanking();
        }
    }

    private void computeRanking() {
        Map<Integer, Double> ig = ranker.rank(originalHeader, warmup);
        List<Map.Entry<Integer, Double>> sorted = new ArrayList<>(ig.entrySet());
        sorted.sort(Comparator.<Map.Entry<Integer, Double>>comparingDouble(Map.Entry::getValue).reversed());
        int k = Math.min(topK, sorted.size());
        int[] picked = new int[k];
        selectedSet = new LinkedHashSet<>();
        for (int i = 0; i < k; i++) {
            picked[i] = sorted.get(i).getKey();
            selectedSet.add(picked[i]);
        }
        java.util.Arrays.sort(picked);
        this.selectedSorted = picked;
        this.filtered = InstanceFilter.createFilteredHeader(originalHeader, picked);
        this.ready = true;
        this.warmup.clear();
    }

    @Override
    public Instances filteredHeader() {
        if (!ready) {
            forceWarmupHeader();
        }
        return filtered;
    }

    private void forceWarmupHeader() {
        int classIdx = originalHeader.classIndex();
        int n = originalHeader.numAttributes();
        int k = Math.min(topK, n - 1);
        int[] picked = new int[k];
        int p = 0;
        for (int i = 0; i < n && p < k; i++) {
            if (i == classIdx) continue;
            picked[p++] = i;
        }
        Arrays.sort(picked);
        this.selectedSorted = picked;
        this.selectedSet = new LinkedHashSet<>();
        for (int x : picked) selectedSet.add(x);
        this.filtered = InstanceFilter.createFilteredHeader(originalHeader, picked);
        this.ready = true;
    }

    @Override
    public Set<Integer> getSelectedIndices() {
        return selectedSet;
    }

    public void finishWarmup() {
        if (ready) {
            return;
        }
        if (warmup.isEmpty()) {
            forceWarmupHeader();
        } else {
            computeRanking();
        }
    }

    @Override
    public String getName() {
        return "StaticTopK[k=" + topK + ",warmup=" + warmupSize + "]";
    }

    public boolean isReady() {
        return ready;
    }

    private void validateSelection(Instances header) {
        int classIdx = header.classIndex();
        for (int idx : selectedSorted) {
            if (idx < 0 || idx >= header.numAttributes()) {
                throw new IllegalStateException(
                        "Pre-warmed feature index " + idx
                                + " is outside the new header");
            }
            if (idx == classIdx) {
                throw new IllegalStateException(
                        "Pre-warmed selection contains the class attribute");
            }
        }
    }
}
