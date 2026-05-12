public class SimulationState {
    private final int size;
    private final double[] x;
    private final double[] y;
    private final double[] vx;
    private final double[] vy;
    private final double[] ax;
    private final double[] ay;
    private final boolean[] isFresh;
    private final boolean[] inContactObstacle;
    private final boolean[] inContactWall;

    public SimulationState(int size) {
        this.size = size;
        this.x = new double[size];
        this.y = new double[size];
        this.vx = new double[size];
        this.vy = new double[size];
        this.ax = new double[size];
        this.ay = new double[size];
        this.isFresh = new boolean[size];
        this.inContactObstacle = new boolean[size];
        this.inContactWall = new boolean[size];
    }

    public int getSize() {
        return size;
    }

    public double[] getX() {
        return x;
    }

    public double[] getY() {
        return y;
    }

    public double[] getVx() {
        return vx;
    }

    public double[] getVy() {
        return vy;
    }

    public double[] getAx() {
        return ax;
    }

    public double[] getAy() {
        return ay;
    }

    public boolean[] getIsFresh() {
        return isFresh;
    }

    public boolean[] getInContactObstacle() {
        return inContactObstacle;
    }

    public boolean[] getInContactWall() {
        return inContactWall;
    }
}
