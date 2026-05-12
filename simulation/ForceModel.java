public class ForceModel {
    public static void computeAccelerations(SimulationConfig config, SimulationState state) {
        int n = state.getSize();
        double[] ax = state.getAx();
        double[] ay = state.getAy();
        double[] x = state.getX();
        double[] y = state.getY();

        for (int i = 0; i < n; i++) {
            ax[i] = 0.0;
            ay[i] = 0.0;
        }

        double kOverM = config.getK() / config.getM();
        double radius = config.getR();
        double r0 = config.getR0();
        double outerRadius = config.getL() / 2.0;

        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double yi = y[i];

            for (int j = i + 1; j < n; j++) {
                double dx = xi - x[j];
                double dy = yi - y[j];
                double dist2 = dx * dx + dy * dy;
                double minDist = 2.0 * radius;
                if (dist2 < minDist * minDist) {
                    double dist = Math.sqrt(dist2);
                    double nx = dist > 0.0 ? dx / dist : 1.0;
                    double ny = dist > 0.0 ? dy / dist : 0.0;
                    double overlap = minDist - dist;
                    double accel = kOverM * overlap;
                    ax[i] += accel * nx;
                    ay[i] += accel * ny;
                    ax[j] -= accel * nx;
                    ay[j] -= accel * ny;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double yi = y[i];
            double dist = Math.sqrt(xi * xi + yi * yi);
            if (dist <= 0.0) {
                continue;
            }
            double nx = xi / dist;
            double ny = yi / dist;

            double overlapObstacle = radius + r0 - dist;
            if (overlapObstacle > 0.0) {
                double accel = kOverM * overlapObstacle;
                ax[i] += accel * nx;
                ay[i] += accel * ny;
            }

            double overlapWall = radius + dist - outerRadius;
            if (overlapWall > 0.0) {
                double accel = kOverM * overlapWall;
                ax[i] -= accel * nx;
                ay[i] -= accel * ny;
            }
        }
    }
}
