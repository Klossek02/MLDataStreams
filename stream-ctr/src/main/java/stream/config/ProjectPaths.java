package stream.config;

import java.nio.file.Files;
import java.nio.file.Path;

public final class ProjectPaths {
    private static final Path PROJECT_ROOT = findProjectRoot();

    private ProjectPaths() {
    }

    public static Path projectRoot() {
        return PROJECT_ROOT;
    }

    public static String avazuExtendedArff() {
        return projectRoot().resolve("avazu/data/avazu_extended.arff").toString();
    }

    public static String avazuHashedArff() {
        Path hashed = projectRoot().resolve("avazu/data/avazu_hashed_100.arff");
        if (Files.exists(hashed)) {
            return hashed.toString();
        }
        return projectRoot().resolve("avazu/data/avazu_extended.arff").toString();
    }

    public static String criteoExtendedArff() {
        return projectRoot().resolve("criteo/data/criteo_extended.arff").toString();
    }

    public static String criteoHashedArff() {
        Path hashed = projectRoot().resolve("criteo/data/criteo_100.arff");
        if (Files.exists(hashed)) {
            return hashed.toString();
        }
        return projectRoot().resolve("criteo/data/criteo_extended.arff").toString();
    }

    public static Path resultsDir() {
        return projectRoot().resolve("results");
    }

    private static Path findProjectRoot() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
        while (current != null) {
            if (Files.isDirectory(current.resolve("avazu"))
                    && Files.isDirectory(current.resolve("criteo"))
                    && Files.isDirectory(current.resolve("stream-ctr"))) {
                return current;
            }
            current = current.getParent();
        }
        return Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    }
}
