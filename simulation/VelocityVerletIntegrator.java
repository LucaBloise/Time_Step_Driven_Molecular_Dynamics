public class VelocityVerletIntegrator {
    public static void step(SimulationConfig config, SimulationState state) {
        double dt = config.getDt();
        double halfDt = 0.5 * dt;
        double dt2 = dt * dt;
        int n = state.getSize();

        double[] x = state.getX();
        double[] y = state.getY();
        double[] vx = state.getVx();
        double[] vy = state.getVy();
        double[] ax = state.getAx();
        double[] ay = state.getAy();

        for (int i = 0; i < n; i++) {
            x[i] += vx[i] * dt + 0.5 * ax[i] * dt2;
            y[i] += vy[i] * dt + 0.5 * ay[i] * dt2;
            vx[i] += ax[i] * halfDt;
            vy[i] += ay[i] * halfDt;
        }

        ForceModel.computeAccelerations(config, state);

        for (int i = 0; i < n; i++) {
            vx[i] += ax[i] * halfDt;
            vy[i] += ay[i] * halfDt;
        }
    }
}
