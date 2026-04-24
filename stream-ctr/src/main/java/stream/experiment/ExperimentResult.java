package stream.experiment;

import stream.drift.DriftEvent;
import stream.evaluation.EvalRecord;

import java.util.List;
import java.util.Map;

public record ExperimentResult(
        String name,
        String datasetName,
        String modelName,
        String selectorName,
        long instances,
        long elapsedMs,
        Map<String, Double> finalMetrics,
        List<EvalRecord> history,
        List<DriftEvent> driftEvents) {}