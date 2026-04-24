package stream.provider;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.util.HashMap;
import java.util.Map;

public class SmokeTestProviders {

    public static void main(String[] args) {
        String arffPath = (args.length > 0)
                ? args[0]
                : "/home/kubog/MLDataStreams/avazu/data/avazu_hashed_100.arff";

        System.out.println("=== Test 1: ArffStreamProvider (Avazu) ===");
        StreamProvider arff = new ArffStreamProvider(arffPath);
        describeAndConsume(arff, 5000);

        System.out.println();
        System.out.println("=== Test 2: ArffStreamProvider restart ===");
        arff.restart();
        int afterRestart = 0;
        while (arff.hasNext() && afterRestart < 10) {
            arff.next();
            afterRestart++;
        }
        System.out.println("After restart consumed: " + afterRestart + " instances");

        System.out.println();
        System.out.println("=== Test 3: AgrawalStreamProvider (abrupt drift) ===");
        AgrawalStreamProvider agrawal = new AgrawalStreamProvider(
                10_000, 5_000, 1, 3);
        describeAndConsume(agrawal, 10_000);

        System.out.println();
        System.out.println("=== Test 4: AgrawalStreamProvider (gradual drift) ===");
        AgrawalStreamProvider agrawalGradual = new AgrawalStreamProvider(
                10_000, 5_000, 2_000, 1, 3, 42);
        describeAndConsume(agrawalGradual, 10_000);

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void describeAndConsume(StreamProvider provider, int limit) {
        Instances header = provider.getHeader();
        System.out.println("Relation: " + header.getRelationName());
        System.out.println("Attributes: " + header.numAttributes()
                + " | classIndex=" + header.classIndex()
                + " | numClasses=" + header.numClasses());

        Map<Double, Integer> dist = new HashMap<>();
        int count = 0;
        long t0 = System.currentTimeMillis();
        while (provider.hasNext() && count < limit) {
            Instance inst = provider.next();
            if (inst.numAttributes() != header.numAttributes()) {
                throw new IllegalStateException(
                        "Instance arity mismatch at " + count);
            }
            dist.merge(inst.classValue(), 1, Integer::sum);
            count++;
        }
        long t1 = System.currentTimeMillis();
        System.out.println("Consumed: " + count + " instances in " + (t1 - t0) + " ms");
        System.out.println("Class distribution: " + dist);
    }
}