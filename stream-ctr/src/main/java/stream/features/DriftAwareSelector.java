package stream.features;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DriftAwareSelector implements FeatureSelector {

    public interface AdaptationListener {
        void onAdaptation(long instanceIndex,
                          Instances newFilteredHeader,
                          Set<Integer> newSelected,
                          Set<Integer> removed,
                          Set<Integer> added,
                          Map<Integer, Double> deltaIg);
    }

    public record AdaptationEvent(long instanceIndex,
                                  Set<Integer> removed,
                                  Set<Integer> added,
                                  Map<Integer, Double> deltaIg) {}

    private final int topK;
    private final int windowSize;
    private final double changeThreshold;
    private final InfoGainRanker ranker;

    private Instances originalHeader;
    private Instances filtered;

    private final Deque<Instance> recentBuffer = new ArrayDeque<>();
    private List<Instance> preDriftSnapshot;
    private Map<Integer, Double> igBefore;

    private boolean awaitingPostDrift;
    private long postDriftCollected;

    private int[] selectedSorted;
    private Set<Integer> selectedSet;

    private long seen;
    private AdaptationListener listener;
    private final List<AdaptationEvent> events = new ArrayList<>();

    public DriftAwareSelector(int topK, int windowSize, double changeThreshold) {
        this(topK, windowSize, changeThreshold, new InfoGainRanker());
    }

    public DriftAwareSelector(int topK, int windowSize, double changeThreshold,
                              InfoGainRanker ranker) {
        if (topK <= 0) throw new IllegalArgumentException("topK > 0");
        if (windowSize <= 0) throw new IllegalArgumentException("windowSize > 0");
        this.topK = topK;
        this.windowSize = windowSize;
        this.changeThreshold = changeThreshold;
        this.ranker = ranker;
    }

    public DriftAwareSelector withListener(AdaptationListener l) {
        this.listener = l;
        return this;
    }

    public List<AdaptationEvent> events() {
        return events;
    }

    @Override
    public void initialize(Instances header) {
        this.originalHeader = header;
        this.recentBuffer.clear();
        this.preDriftSnapshot = null;
        this.igBefore = null;
        this.awaitingPostDrift = false;
        this.postDriftCollected = 0;
        this.seen = 0;
        this.events.clear();

        int classIdx = header.classIndex();
        int n = header.numAttributes();
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
        this.filtered = InstanceFilter.createFilteredHeader(header, picked);
    }

    @Override
    public Instance filter(Instance instance) {
        return InstanceFilter.filterAttributes(instance, selectedSorted, filtered);
    }

    @Override
    public void observe(Instance instance, int trueLabel) {
        seen++;
        recentBuffer.addLast(instance);
        if (recentBuffer.size() > windowSize) recentBuffer.removeFirst();

        if (awaitingPostDrift) {
            postDriftCollected++;
            if (postDriftCollected >= windowSize) {
                adaptFeatures();
                awaitingPostDrift = false;
                postDriftCollected = 0;
                preDriftSnapshot = null;
                igBefore = null;
            }
        }
    }

    public void onDriftDetected() {
        if (recentBuffer.size() < Math.max(2, windowSize / 4)) {
            return;
        }
        preDriftSnapshot = new ArrayList<>(recentBuffer);
        igBefore = ranker.rank(originalHeader, preDriftSnapshot);
        recentBuffer.clear();
        awaitingPostDrift = true;
        postDriftCollected = 0;
    }

    private void adaptFeatures() {
        List<Instance> postSnapshot = new ArrayList<>(recentBuffer);
        Map<Integer, Double> igAfter = ranker.rank(originalHeader, postSnapshot);

        Map<Integer, Double> delta = new HashMap<>();
        for (Map.Entry<Integer, Double> e : igAfter.entrySet()) {
            double before = igBefore.getOrDefault(e.getKey(), 0.0);
            delta.put(e.getKey(), e.getValue() - before);
        }

        List<Map.Entry<Integer, Double>> ranked = new ArrayList<>(igAfter.entrySet());
        ranked.sort(Comparator.<Map.Entry<Integer, Double>>comparingDouble(Map.Entry::getValue).reversed());

        Set<Integer> baseTopK = new LinkedHashSet<>();
        int kCap = Math.min(topK, ranked.size());
        for (int i = 0; i < kCap; i++) baseTopK.add(ranked.get(i).getKey());

        Set<Integer> finalSet = new LinkedHashSet<>(baseTopK);
        for (Map.Entry<Integer, Double> e : delta.entrySet()) {
            int f = e.getKey();
            double d = e.getValue();
            if (d > changeThreshold && !finalSet.contains(f)) {
                finalSet.add(f);
            }
        }
        for (Map.Entry<Integer, Double> e : delta.entrySet()) {
            int f = e.getKey();
            double d = e.getValue();
            if (d < -changeThreshold) {
                finalSet.remove(f);
            }
        }
        if (finalSet.size() > topK) {
            List<Integer> shrink = new ArrayList<>(finalSet);
            shrink.sort(Comparator.comparingDouble(a -> -igAfter.getOrDefault(a, 0.0)));
            finalSet = new LinkedHashSet<>(shrink.subList(0, topK));
        } else if (finalSet.isEmpty()) {
            finalSet = baseTopK;
        }

        Set<Integer> removed = new LinkedHashSet<>(selectedSet);
        removed.removeAll(finalSet);
        Set<Integer> added = new LinkedHashSet<>(finalSet);
        added.removeAll(selectedSet);

        int[] picked = finalSet.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(picked);
        this.selectedSorted = picked;
        this.selectedSet = new LinkedHashSet<>();
        for (int x : picked) selectedSet.add(x);
        this.filtered = InstanceFilter.createFilteredHeader(originalHeader, picked);

        AdaptationEvent ev = new AdaptationEvent(seen, removed, added, delta);
        events.add(ev);
        if (listener != null) {
            listener.onAdaptation(seen, filtered, selectedSet, removed, added, delta);
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
        return "DriftAware[k=" + topK + ",win=" + windowSize
                + ",thr=" + changeThreshold + "]";
    }
}