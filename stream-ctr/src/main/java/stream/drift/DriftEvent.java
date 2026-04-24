package stream.drift;

public record DriftEvent(
        long instanceIndex,
        long timestamp,
        int windowSizeBefore,
        int windowSizeAfter,
        String detectorName
) {}