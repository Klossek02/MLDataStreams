package stream.config;

public record DatasetInfo(
        String name,
        int numInstances,
        int numAttributes,
        int numClasses,
        String[] attributeNames
) {}