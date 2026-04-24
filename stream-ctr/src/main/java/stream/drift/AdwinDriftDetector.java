package stream.drift;

import moa.classifiers.core.driftdetection.ADWIN;

public class AdwinDriftDetector implements DriftDetector {

    private final double delta;
    private ADWIN adwin;

    public AdwinDriftDetector() {
        this(0.002);
    }

    public AdwinDriftDetector(double delta) {
        this.delta = delta;
        this.adwin = new ADWIN(delta);
    }

    @Override
    public boolean detect(double value) {
        return adwin.setInput(value);
    }

    @Override
    public boolean isWarning() {
        return false;
    }

    @Override
    public int getWindowSize() {
        return (int) adwin.getWidth();
    }

    @Override
    public void reset() {
        this.adwin = new ADWIN(delta);
    }

    @Override
    public String getName() {
        return "ADWIN(delta=" + delta + ")";
    }
}