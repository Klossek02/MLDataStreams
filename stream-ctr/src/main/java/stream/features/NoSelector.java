package stream.features;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.LinkedHashSet;
import java.util.Set;

public class NoSelector implements FeatureSelector {

    private Instances header;
    private Set<Integer> indices;

    @Override
    public void initialize(Instances header) {
        this.header = header;
        this.indices = new LinkedHashSet<>();
        int classIdx = header.classIndex();
        for (int i = 0; i < header.numAttributes(); i++) {
            if (i != classIdx) indices.add(i);
        }
    }

    @Override
    public Instance filter(Instance instance) {
        return instance;
    }

    @Override
    public void observe(Instance instance, int trueLabel) {}

    @Override
    public Instances filteredHeader() {
        return header;
    }

    @Override
    public Set<Integer> getSelectedIndices() {
        return indices;
    }

    @Override
    public String getName() {
        return "NoSelector";
    }
}