import java.io.BufferedWriter;
import java.io.IOException;

public class SimulationRunner {
    public static void run(SimulationConfig config, SimulationState state, BufferedWriter writer, BufferedWriter events)
            throws IOException {
        int steps = (int) Math.ceil(config.getTf() / config.getDt());
        double t = 0.0;
        double nextOutputTime = 0.0;

        for (int step = 0; step <= steps; step++) {
            if (t + 1e-12 >= nextOutputTime) {
                OutputWriter.writeState(writer, state, t);
                nextOutputTime += config.getDt2();
            }
            if (step == steps) {
                break;
            }

            VelocityVerletIntegrator.step(config, state);
            t += config.getDt();
            EventTracker.updateStates(config, state, t, events);
        }
    }
}
