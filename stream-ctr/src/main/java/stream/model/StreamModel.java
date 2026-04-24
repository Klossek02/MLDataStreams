package stream.model;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public interface StreamModel {

    void initialize(Instances header);

    double predictProbability(Instance instance);

    void train(Instance instance);

    void reset();

    String getName();
}