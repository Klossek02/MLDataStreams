package stream.experiment;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import stream.features.DriftAwareSelector;
import stream.provider.StreamProvider;

public class ObservingProvider implements StreamProvider {

    private final StreamProvider inner;
    private final DriftAwareSelector selector;

    public ObservingProvider(StreamProvider inner, DriftAwareSelector selector) {
        this.inner = inner;
        this.selector = selector;
    }

    @Override
    public Instances getHeader() {
        return inner.getHeader();
    }

    @Override
    public boolean hasNext() {
        return inner.hasNext();
    }

    @Override
    public Instance next() {
        Instance ins = inner.next();
        selector.observe(ins, (int) ins.classValue());
        return ins;
    }

    @Override
    public void restart() {
        inner.restart();
    }
}