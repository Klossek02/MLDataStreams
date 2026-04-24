package stream.drift;

public interface DriftDetector {

    boolean detect(double value);

    boolean isWarning();

    int getWindowSize();

    void reset();

    String getName();
}