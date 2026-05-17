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
        double minDist = 2.0 * radius;
        double minDist2 = minDist * minDist;

        // Cell Index Method: same contact law, faster neighbor search.
        double domainLength = config.getL();
        int cellsPerSide = Math.max(1, (int) Math.floor(domainLength / minDist));
        double cellSize = domainLength / cellsPerSide;
        int cellCount = cellsPerSide * cellsPerSide;

        int[] head = new int[cellCount];
        int[] next = new int[n];
        int[] cellX = new int[n];
        int[] cellY = new int[n];

        for (int i = 0; i < cellCount; i++) {
            head[i] = -1;
        }

        for (int i = 0; i < n; i++) {
            int cx = toCellIndex(x[i], outerRadius, cellSize, cellsPerSide);
            int cy = toCellIndex(y[i], outerRadius, cellSize, cellsPerSide);
            cellX[i] = cx;
            cellY[i] = cy;

            int cellId = cy * cellsPerSide + cx;
            next[i] = head[cellId];
            head[cellId] = i;
        }

        for (int i = 0; i < n; i++) {
            double xi = x[i];
            double yi = y[i];
            int cx = cellX[i];
            int cy = cellY[i];

            for (int oy = -1; oy <= 1; oy++) {
                int nyCell = cy + oy;
                if (nyCell < 0 || nyCell >= cellsPerSide) {
                    continue;
                }

                for (int ox = -1; ox <= 1; ox++) {
                    int nxCell = cx + ox;
                    if (nxCell < 0 || nxCell >= cellsPerSide) {
                        continue;
                    }

                    int cellId = nyCell * cellsPerSide + nxCell;
                    for (int j = head[cellId]; j != -1; j = next[j]) {
                        if (j <= i) {
                            continue;
                        }

                        double dx = xi - x[j];
                        double dy = yi - y[j];
                        double dist2 = dx * dx + dy * dy;
                        if (dist2 < minDist2) {
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

    private static int toCellIndex(double coord, double outerRadius, double cellSize, int cellsPerSide) {
        double shifted = coord + outerRadius;
        int idx = (int) Math.floor(shifted / cellSize);
        if (idx < 0) {
            return 0;
        }
        if (idx >= cellsPerSide) {
            return cellsPerSide - 1;
        }
        return idx;
    }
}
