package stream.model;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import stream.config.ProjectPaths;
import stream.provider.AgrawalStreamProvider;
import stream.provider.ArffStreamProvider;
import stream.provider.StreamProvider;

import java.util.List;
import java.util.function.Supplier;

public class SmokeTestAllModels {

    public static void main(String[] args) {
        String arffPath = (args.length > 0)
                ? args[0]
                : ProjectPaths.avazuHashedArff();

        List<Supplier<StreamModel>> models = List.of(
                HoeffdingTreeModel::new,
                HatModel::new,
                SrpModel::new
        );

        System.out.println("=== Agrawal 10k (drift @ 5k, fn 1 -> 3) ===");
        for (Supplier<StreamModel> sup : models) {
            StreamProvider p = new AgrawalStreamProvider(10_000, 5_000, 1, 3);
            runPrequential(sup.get(), p, 10_000);
        }

        System.out.println();
        System.out.println("=== Avazu 10k ===");
        for (Supplier<StreamModel> sup : models) {
            StreamProvider p = new ArffStreamProvider(arffPath);
            runPrequential(sup.get(), p, 10_000);
        }

        System.out.println();
        System.out.println("=== Probability sanity check (each model returns p in [0,1]) ===");
        for (Supplier<StreamModel> sup : models) {
            StreamModel m = sup.get();
            StreamProvider p = new AgrawalStreamProvider(500, 500, 1, 1);
            m.initialize(p.getHeader());
            double minP = 1.0, maxP = 0.0;
            int n = 0;
            while (p.hasNext()) {
                Instance inst = p.next();
                double prob = m.predictProbability(inst);
                if (prob < 0.0 || prob > 1.0 || Double.isNaN(prob)) {
                    throw new AssertionError(m.getName() + " produced invalid p=" + prob);
                }
                if (prob < minP) minP = prob;
                if (prob > maxP) maxP = prob;
                m.train(inst);
                n++;
            }
            System.out.printf("%-16s n=%d minP=%.4f maxP=%.4f%n",
                    m.getName(), n, minP, maxP);
        }

        System.out.println();
        System.out.println("All smoke tests passed.");
    }

    private static void runPrequential(StreamModel model,
                                       StreamProvider provider,
                                       int limit) {
        Instances header = provider.getHeader();
        model.initialize(header);

        int n = 0;
        int correct = 0;
        double logLossSum = 0.0;
        long t0 = System.currentTimeMillis();

        while (provider.hasNext() && n < limit) {
            Instance inst = provider.next();
            double p = model.predictProbability(inst);
            int yHat = (p >= 0.5) ? 1 : 0;
            int y = (int) inst.classValue();
            if (yHat == y) correct++;
            double pc = Math.max(1e-15, Math.min(1.0 - 1e-15, p));
            logLossSum += -(y * Math.log(pc) + (1 - y) * Math.log(1 - pc));
            model.train(inst);
            n++;
        }

        long t1 = System.currentTimeMillis();
        System.out.printf("%-16s n=%d acc=%.4f logloss=%.4f time=%d ms%n",
                model.getName(), n, correct / (double) n, logLossSum / n, (t1 - t0));
    }
}
