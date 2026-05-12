import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Initializer {
    public static SimulationState initialize(SimulationConfig config) {
        int n = config.getN();
        SimulationState state = new SimulationState(n);
        Random random = new Random(config.getSeed());

        double outerRadius = config.getL() / 2.0 - config.getR();
        double innerRadius = config.getR0() + config.getR();

        double[] x = state.getX();
        double[] y = state.getY();

        boolean placedRandomly = placeRandomly(config, x, y, outerRadius, innerRadius, random);
        if (!placedRandomly) {
            placeOnLattice(config, x, y, outerRadius, innerRadius, random);
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

    private static boolean placeRandomly(SimulationConfig config,
                                         double[] x,
                                         double[] y,
                                         double outerRadius,
                                         double innerRadius,
                                         Random random) {
        int n = config.getN();
        int maxAttempts = Math.max(20000, 50 * n);
        for (int i = 0; i < n; i++) {
            boolean placed = false;
            for (int attempt = 0; attempt < maxAttempts; attempt++) {
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
                return false;
            }
        }
        return true;
    }

    private static void placeOnLattice(SimulationConfig config,
                                       double[] x,
                                       double[] y,
                                       double outerRadius,
                                       double innerRadius,
                                       Random random) {
        double r = config.getR();
        double dx = 2.0 * r;
        double dy = Math.sqrt(3.0) * r;

        List<Position> positions = new ArrayList<>();
        int row = 0;
        for (double yy = -outerRadius; yy <= outerRadius; yy += dy) {
            double offset = (row % 2 == 0) ? 0.0 : r;
            for (double xx = -outerRadius + offset; xx <= outerRadius; xx += dx) {
                double dist = Math.sqrt(xx * xx + yy * yy);
                if (dist >= innerRadius && dist <= outerRadius) {
                    positions.add(new Position(xx, yy));
                }
            }
            row++;
        }

        if (positions.size() < config.getN()) {
            throw new IllegalStateException("Not enough lattice positions for N=" + config.getN());
        }

        Collections.shuffle(positions, random);
        for (int i = 0; i < config.getN(); i++) {
            Position p = positions.get(i);
            x[i] = p.x;
            y[i] = p.y;
        }
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

    private static final class Position {
        final double x;
        final double y;

        Position(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }
}
