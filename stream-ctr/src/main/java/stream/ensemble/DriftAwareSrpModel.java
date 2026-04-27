package stream.ensemble;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import stream.features.InstanceFilter;
import stream.model.HoeffdingTreeModel;
import stream.model.StreamModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

public class DriftAwareSrpModel implements StreamModel {

    private final String name;
    private final int ensembleSize;
    private final int subspaceSize;
    private final long seed;
    private final Supplier<StreamModel> baseFactory;

    private final List<StreamModel> baseModels = new ArrayList<>();
    private final List<int[]> sortedSubspaces = new ArrayList<>();
    private final List<Instances> subspaceHeaders = new ArrayList<>();
    private SubspaceManager subspaceManager;
    private WeightManager weightManager;

    private Instances originalHeader;
    private Random rng;

    private int minWeakOverlap = 1;
    private double weightResetValue = 0.3;
    private double weightDecayRate = 0.001;
    private double resetRatio = 0.0;
    private int cooldownInstances = 0;
    private long instancesSeen = 0;
    private long lastAdaptationAt;

    public DriftAwareSrpModel withMinWeakOverlap(int n) {
        if (n < 1) throw new IllegalArgumentException("minWeakOverlap >= 1");
        this.minWeakOverlap = n;
        return this;
    }

    public DriftAwareSrpModel withWeightRecovery(double resetValue, double decayRate) {
        this.weightResetValue = resetValue;
        this.weightDecayRate = decayRate;
        return this;
    }

    public DriftAwareSrpModel withResetRatio(double r) {
        if (r < 0 || r > 1) throw new IllegalArgumentException("resetRatio in [0,1]");
        this.resetRatio = r;
        return this;
    }

    public DriftAwareSrpModel withAdaptationCooldown(int instances) {
        if (instances < 0) throw new IllegalArgumentException("cooldown >= 0");
        this.cooldownInstances = instances;
        return this;
    }

    public DriftAwareSrpModel() {
        this("DriftAwareSRP", 10, 5, 42L, HoeffdingTreeModel::new);
    }

    public DriftAwareSrpModel(String name,
                              int ensembleSize,
                              int subspaceSize,
                              long seed,
                              Supplier<StreamModel> baseFactory) {
        this.name = name;
        this.ensembleSize = ensembleSize;
        this.subspaceSize = subspaceSize;
        this.seed = seed;
        this.baseFactory = baseFactory;
    }

    @Override
    public void initialize(Instances header) {
        this.originalHeader = header;
        this.rng = new Random(seed);
        this.subspaceManager = new SubspaceManager(ensembleSize, subspaceSize, seed);
        this.subspaceManager.initRandom(header.numAttributes(), header.classIndex());
        this.instancesSeen = 0;
        this.lastAdaptationAt = -cooldownInstances;
        this.weightManager = new WeightManager(ensembleSize, weightResetValue, 1.0, weightDecayRate);

        baseModels.clear();
        sortedSubspaces.clear();
        subspaceHeaders.clear();

        for (int i = 0; i < ensembleSize; i++) {
            int[] sorted = sortedFromSet(subspaceManager.subspace(i));
            Instances subHeader = InstanceFilter.createFilteredHeader(header, sorted);
            StreamModel m = baseFactory.get();
            m.initialize(subHeader);
            baseModels.add(m);
            sortedSubspaces.add(sorted);
            subspaceHeaders.add(subHeader);
        }
    }

    @Override
    public double predictProbability(Instance instance) {
        double[] probs = new double[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) {
            Instance f = InstanceFilter.filterAttributes(
                    instance, sortedSubspaces.get(i), subspaceHeaders.get(i));
            probs[i] = baseModels.get(i).predictProbability(f);
        }
        return weightManager.weightedPrediction(probs);
    }

    @Override
    public void train(Instance instance) {
        for (int i = 0; i < ensembleSize; i++) {
            int k = poisson1();
            if (k <= 0) continue;
            Instance f = InstanceFilter.filterAttributes(
                    instance, sortedSubspaces.get(i), subspaceHeaders.get(i));
            for (int r = 0; r < k; r++) {
                baseModels.get(i).train(f);
            }
        }
        weightManager.decay();
        instancesSeen++;
    }

    public void onDriftDetected(Set<Integer> weakFeatures, Set<Integer> strongFeatures) {
        if (cooldownInstances > 0) {
            long since = instancesSeen - lastAdaptationAt;
            if (since < cooldownInstances) {
                System.out.println("    [cooldown] adaptation skipped (instancesSinceLast="
                        + since + " < " + cooldownInstances + ")");
                return;
            }
        }
        SubspaceManager.AdaptationResult res =
                subspaceManager.adaptSubspaces(weakFeatures, strongFeatures,
                        minWeakOverlap, resetRatio);

        if (res.modelsChanged().isEmpty()) {
            return;
        }

        for (int idx : res.modelsChanged()) {
            int[] sorted = sortedFromSet(subspaceManager.subspace(idx));
            Instances subHeader = InstanceFilter.createFilteredHeader(originalHeader, sorted);
            StreamModel fresh = baseFactory.get();
            fresh.initialize(subHeader);
            baseModels.set(idx, fresh);
            sortedSubspaces.set(idx, sorted);
            subspaceHeaders.set(idx, subHeader);
        }
        if (!res.modelsToReset().isEmpty()) {
            weightManager.onModelsReset(res.modelsToReset());
        }
        lastAdaptationAt = instancesSeen;
    }

    @Override
    public void reset() {
        if (originalHeader != null) initialize(originalHeader);
    }

    @Override
    public String getName() {
        return name;
    }

    public SubspaceManager subspaceManager() {
        return subspaceManager;
    }

    public WeightManager weightManager() {
        return weightManager;
    }

    public List<StreamModel> baseModels() {
        return baseModels;
    }

    private int poisson1() {
        double l = Math.exp(-1.0);
        int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= rng.nextDouble();
        } while (p > l);
        return k - 1;
    }

    private static int[] sortedFromSet(Set<Integer> s) {
        int[] arr = new int[s.size()];
        int i = 0;
        for (int x : s) arr[i++] = x;
        Arrays.sort(arr);
        return arr;
    }

    public static Set<Integer> setOf(int... vals) {
        Set<Integer> s = new LinkedHashSet<>();
        for (int v : vals) s.add(v);
        return s;
    }
}
