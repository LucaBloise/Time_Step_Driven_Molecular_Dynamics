import java.util.Random;

public class Initializer {
    public static SimulationState initialize(SimulationConfig config) {
        int n = config.getN();
        SimulationState state = new SimulationState(n);
        Random random = new Random(config.getSeed());

        double outerRadius = config.getL() / 2.0 - config.getR();
        double innerRadius = config.getR0() + config.getR();

        double[] x = state.getX();
        double[] y = state.getY();

        for (int i = 0; i < n; i++) {
            boolean placed = false;
            for (int attempt = 0; attempt < 10000; attempt++) {
                double u = random.nextDouble();
                double radius = Math.sqrt(u * (outerRadius * outerRadius - innerRadius * innerRadius)
                        + innerRadius * innerRadius);
                double angle = 2.0 * Math.PI * random.nextDouble();
                double nx = radius * Math.cos(angle);
                double ny = radius * Math.sin(angle);
                if (isValidPosition(i, x, y, nx, ny, config.getR())) {
                    x[i] = nx;
                    y[i] = ny;
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                throw new IllegalStateException("Failed to place particle " + i + " without overlap.");
            }
        }

        double[] vx = state.getVx();
        double[] vy = state.getVy();
        boolean[] isFresh = state.getIsFresh();

        for (int i = 0; i < n; i++) {
            double angle = 2.0 * Math.PI * random.nextDouble();
            vx[i] = config.getV0() * Math.cos(angle);
            vy[i] = config.getV0() * Math.sin(angle);
            isFresh[i] = true;
        }

        ForceModel.computeAccelerations(config, state);
        return state;
    }

    private static boolean isValidPosition(int count, double[] x, double[] y, double nx, double ny, double r) {
        double minDist = 2.0 * r;
        double minDist2 = minDist * minDist;
        for (int i = 0; i < count; i++) {
            double dx = nx - x[i];
            double dy = ny - y[i];
            if (dx * dx + dy * dy < minDist2) {
                return false;
            }
        }
        return true;
    }
}
