package stream.features;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class OnlineRankingSelector implements FeatureSelector {

    public interface ReinitListener {
        void onReinitialize(Instances newFilteredHeader, Set<Integer> newSelected);
    }

    private final int topK;
    private final int bufferSize;
    private final int rerankEvery;
    private final InfoGainRanker ranker;

    private Instances originalHeader;
    private Instances filtered;
    private final Deque<Instance> buffer = new ArrayDeque<>();
    private int[] selectedSorted;
    private Set<Integer> selectedSet;
    private long seen;
    private ReinitListener listener;

    public OnlineRankingSelector(int topK, int bufferSize, int rerankEvery) {
        this(topK, bufferSize, rerankEvery, new InfoGainRanker());
    }

    public OnlineRankingSelector(int topK, int bufferSize, int rerankEvery, InfoGainRanker ranker) {
        this.topK = topK;
        this.bufferSize = bufferSize;
        this.rerankEvery = rerankEvery;
        this.ranker = ranker;
    }

    public OnlineRankingSelector withReinitListener(ReinitListener l) {
        this.listener = l;
        return this;
    }

    @Override
    public void initialize(Instances header) {
        this.originalHeader = header;
        this.buffer.clear();
        this.seen = 0;
        int classIdx = header.classIndex();
        int n = header.numAttributes();
        int k = Math.min(topK, n - 1);
        int[] picked = new int[k];
        int p = 0;
        for (int i = 0; i < n && p < k; i++) {
            if (i == classIdx) continue;
            picked[p++] = i;
        }
        java.util.Arrays.sort(picked);
        this.selectedSorted = picked;
        this.selectedSet = new LinkedHashSet<>();
        for (int x : picked) selectedSet.add(x);
        this.filtered = InstanceFilter.createFilteredHeader(header, picked);
    }

    @Override
    public Instance filter(Instance instance) {
        return InstanceFilter.filterAttributes(instance, selectedSorted, filtered);
    }

    @Override
    public void observe(Instance instance, int trueLabel) {
        buffer.addLast(instance);
        if (buffer.size() > bufferSize) buffer.removeFirst();
        seen++;
        if (seen % rerankEvery == 0 && buffer.size() >= Math.min(bufferSize, rerankEvery)) {
            rerank();
        }
    }

    private void rerank() {
        List<Instance> snapshot = new ArrayList<>(buffer);
        Map<Integer, Double> ig = ranker.rank(originalHeader, snapshot);
        List<Map.Entry<Integer, Double>> sorted = new ArrayList<>(ig.entrySet());
        sorted.sort(Comparator.<Map.Entry<Integer, Double>>comparingDouble(Map.Entry::getValue).reversed());
        int k = Math.min(topK, sorted.size());
        int[] picked = new int[k];
        Set<Integer> newSet = new LinkedHashSet<>();
        for (int i = 0; i < k; i++) {
            picked[i] = sorted.get(i).getKey();
            newSet.add(picked[i]);
        }
        java.util.Arrays.sort(picked);

        boolean changed = !newSet.equals(selectedSet);
        this.selectedSorted = picked;
        this.selectedSet = newSet;
        this.filtered = InstanceFilter.createFilteredHeader(originalHeader, picked);
        if (changed && listener != null) {
            listener.onReinitialize(filtered, selectedSet);
        }
    }

    @Override
    public Instances filteredHeader() {
        return filtered;
    }

    @Override
    public Set<Integer> getSelectedIndices() {
        return selectedSet;
    }

    @Override
    public String getName() {
        return "OnlineRanking[k=" + topK + ",buf=" + bufferSize + ",every=" + rerankEvery + "]";
    }
}