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

        // Random sequential placement jams at high densities; use lattice-first for large N.
        boolean useRandomFirst = n <= 700;
        boolean placedRandomly = useRandomFirst && placeRandomly(config, x, y, outerRadius, innerRadius, random);
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

        List<Position> bestPositions = buildHexLatticePositions(outerRadius, innerRadius, dx, dy, 0.0, 0.0);

        // Try multiple lattice offsets and keep the densest valid arrangement.
        for (int attempt = 0; attempt < 32; attempt++) {
            double shiftX = random.nextDouble() * dx;
            double shiftY = random.nextDouble() * dy;
            List<Position> candidate = buildHexLatticePositions(outerRadius, innerRadius, dx, dy, shiftX, shiftY);
            if (candidate.size() > bestPositions.size()) {
                bestPositions = candidate;
            }
        }

        if (bestPositions.size() < config.getN()) {
            throw new IllegalStateException(
                    "Not enough lattice positions for N=" + config.getN() + " (max=" + bestPositions.size() + ")");
        }

        Collections.shuffle(bestPositions, random);
        for (int i = 0; i < config.getN(); i++) {
            Position p = bestPositions.get(i);
            x[i] = p.x;
            y[i] = p.y;
        }
    }

    private static List<Position> buildHexLatticePositions(double outerRadius,
                                                           double innerRadius,
                                                           double dx,
                                                           double dy,
                                                           double shiftX,
                                                           double shiftY) {
        List<Position> positions = new ArrayList<>();
        int row = 0;

        for (double yy = -outerRadius + shiftY; yy <= outerRadius + 1e-12; yy += dy) {
            double rowOffset = (row % 2 == 0) ? 0.0 : dx / 2.0;
            for (double xx = -outerRadius + rowOffset + shiftX; xx <= outerRadius + 1e-12; xx += dx) {
                double dist = Math.sqrt(xx * xx + yy * yy);
                if (dist >= innerRadius && dist <= outerRadius) {
                    positions.add(new Position(xx, yy));
                }
            }
            row++;
        }

        return positions;
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
