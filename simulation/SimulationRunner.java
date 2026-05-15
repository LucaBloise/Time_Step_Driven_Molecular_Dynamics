import java.io.BufferedWriter;
import java.io.IOException;

public class SimulationRunner {
    public static long run(SimulationConfig config, SimulationState state, BufferedWriter writer, BufferedWriter events)
            throws IOException {
        long startTime = System.currentTimeMillis();

        int steps = (int) Math.ceil(config.getTf() / config.getDt());
        double t = 0.0;
        double nextOutputTime = 0.0;

        for (int step = 0; step <= steps; step++) {
            if (t + 1e-12 >= nextOutputTime) {
                if (writer != null && !config.isNoState()) {
                    OutputWriter.writeState(writer, state, t);
                }
                nextOutputTime += config.getDt2();
            }
            if (step == steps) {
                break;
            }

            VelocityVerletIntegrator.step(config, state);
            t += config.getDt();
            if (events != null && !config.isNoEvents()) {
                EventTracker.updateStates(config, state, t, events);
            }
        }

        long endTime = System.currentTimeMillis();
        return endTime - startTime;
    }
}
