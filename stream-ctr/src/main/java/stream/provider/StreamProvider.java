package stream.provider;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

public interface StreamProvider {

    Instances getHeader();

    boolean hasNext();

    Instance next();

    void restart();
}