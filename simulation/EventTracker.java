import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Locale;

public class EventTracker {
    public static void updateStates(SimulationConfig config, SimulationState state, double time, BufferedWriter events)
            throws IOException {
        int n = state.getSize();
        double radius = config.getR();
        double r0 = config.getR0();
        double outerRadius = config.getL() / 2.0;

        double[] x = state.getX();
        double[] y = state.getY();
        boolean[] isFresh = state.getIsFresh();
        boolean[] inContactObstacle = state.getInContactObstacle();
        boolean[] inContactWall = state.getInContactWall();

        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double yi = y[i];
            double dist = Math.sqrt(xi * xi + yi * yi);

            double overlapObstacle = radius + r0 - dist;
            boolean contactObstacle = overlapObstacle > 0.0;
            if (contactObstacle && isFresh[i] && !inContactObstacle[i]) {
                isFresh[i] = false;
                inContactObstacle[i] = true;
                writeEvent(events, time, i, "FRESH_TO_USED");
            } else if (!contactObstacle) {
                inContactObstacle[i] = false;
            }

            double overlapWall = radius + dist - outerRadius;
            boolean contactWall = overlapWall > 0.0;
            if (contactWall && !isFresh[i] && !inContactWall[i]) {
                isFresh[i] = true;
                inContactWall[i] = true;
                writeEvent(events, time, i, "USED_TO_FRESH");
            } else if (!contactWall) {
                inContactWall[i] = false;
            }
        }
    }

    private static void writeEvent(BufferedWriter writer, double time, int id, String event) throws IOException {
        writer.write(String.format(Locale.US, "%.8f %d %s\n", time, id, event));
    }
}
