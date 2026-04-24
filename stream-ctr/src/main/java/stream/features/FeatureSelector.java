package stream.features;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.Set;

public interface FeatureSelector {

    void initialize(Instances header);

    Instance filter(Instance instance);

    void observe(Instance instance, int trueLabel);

    Instances filteredHeader();

    Set<Integer> getSelectedIndices();

    String getName();
}