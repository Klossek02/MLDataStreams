package stream.features;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import java.util.ArrayList;
import java.util.List;

public final class InstanceFilter {

    private InstanceFilter() {}

    public static Instances createFilteredHeader(Instances original, int[] sortedIndices) {
        int classIdx = original.classIndex();
        List<Attribute> attrs = new ArrayList<>(sortedIndices.length + 1);
        for (int idx : sortedIndices) {
            if (idx == classIdx) continue;
            attrs.add(original.attribute(idx));
        }
        attrs.add(original.attribute(classIdx));

        Instances h = new Instances(original.getRelationName() + "_filtered", attrs, 0);
        h.setClassIndex(attrs.size() - 1);
        return new InstancesHeader(h);
    }

    public static Instance filterAttributes(Instance original,
                                            int[] sortedIndices,
                                            Instances filteredHeader) {
        int classIdx = original.classIndex();
        int newSize = filteredHeader.numAttributes();
        double[] values = new double[newSize];

        int outIdx = 0;
        for (int idx : sortedIndices) {
            if (idx == classIdx) continue;
            values[outIdx++] = original.value(idx);
        }
        values[newSize - 1] = original.classValue();

        Instance ni = new DenseInstance(original.weight(), values);
        ni.setDataset(filteredHeader);
        return ni;
    }
}