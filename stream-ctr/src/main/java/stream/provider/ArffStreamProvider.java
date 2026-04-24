package stream.provider;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.streams.ArffFileStream;

import java.io.File;

public class ArffStreamProvider implements StreamProvider {

    private final String path;
    private final int classIndex;
    private ArffFileStream stream;

    public ArffStreamProvider(String path) {
        this(path, -1);
    }

    public ArffStreamProvider(String path, int classIndex) {
        File f = new File(path);
        if (!f.exists() || !f.isFile()) {
            throw new IllegalArgumentException("ARFF file not found: " + path);
        }
        this.path = path;
        this.classIndex = classIndex;
        init();
    }

    private void init() {
        this.stream = new ArffFileStream(path, classIndex);
        this.stream.prepareForUse();
    }

    @Override
    public Instances getHeader() {
        return stream.getHeader();
    }

    @Override
    public boolean hasNext() {
        return stream.hasMoreInstances();
    }

    @Override
    public Instance next() {
        return stream.nextInstance().getData();
    }

    @Override
    public void restart() {
        stream.restart();
    }
}