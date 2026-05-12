import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

public class SimulationConfig {
    private int n = 200;
    private double l = 80.0;
    private double r0 = 1.0;
    private double r = 1.0;
    private double m = 1.0;
    private double k = 1.0e3;
    private double v0 = 1.0;
    private double tf = 500.0;
    private double dt = 0.001;
    private double dt2 = -1.0;
    private long seed = -1L;
    private String outPath = null;
    private String eventsPath = null;

    public static SimulationConfig fromArgs(String[] args) {
        SimulationConfig config = new SimulationConfig();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--help".equals(arg) || "-h".equals(arg)) {
                printUsageAndExit();
            }
            if (!arg.startsWith("--")) {
                continue;
            }
            String key = arg.substring(2);
            if (i + 1 >= args.length) {
                throw new IllegalArgumentException("Missing value for --" + key);
            }
            String value = args[++i];
            switch (key) {
                case "n" -> config.n = Integer.parseInt(value);
                case "l" -> config.l = Double.parseDouble(value);
                case "r0" -> config.r0 = Double.parseDouble(value);
                case "r" -> config.r = Double.parseDouble(value);
                case "m" -> config.m = Double.parseDouble(value);
                case "k" -> config.k = Double.parseDouble(value);
                case "v0" -> config.v0 = Double.parseDouble(value);
                case "tf" -> config.tf = Double.parseDouble(value);
                case "dt" -> config.dt = Double.parseDouble(value);
                case "dt2" -> config.dt2 = Double.parseDouble(value);
                case "seed" -> config.seed = Long.parseLong(value);
                case "out" -> config.outPath = value;
                case "events-out" -> config.eventsPath = value;
                default -> throw new IllegalArgumentException("Unknown option: --" + key);
            }
        }
        return config;
    }

    public void applyDefaults() {
        if (seed < 0) {
            seed = System.nanoTime();
        }
        if (dt2 <= 0.0) {
            dt2 = dt * 10.0;
        }
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String baseName = String.format(Locale.US, "scan_N%d_k%.0f_seed%d_%s", n, k, seed, timestamp);
        if (outPath == null || outPath.isBlank()) {
            outPath = "outputs/scanningRate/" + baseName + ".txt";
        }
        if (eventsPath == null || eventsPath.isBlank()) {
            eventsPath = "outputs/scanningRate/" + baseName + "_events.txt";
        }
    }

    public void validate() {
        if (n <= 0) {
            throw new IllegalArgumentException("N must be positive.");
        }
        if (l <= 0.0 || r <= 0.0 || r0 <= 0.0) {
            throw new IllegalArgumentException("L, r, and r0 must be positive.");
        }
        if (m <= 0.0 || k <= 0.0) {
            throw new IllegalArgumentException("m and k must be positive.");
        }
        if (dt <= 0.0 || tf <= 0.0) {
            throw new IllegalArgumentException("dt and tf must be positive.");
        }
        double outerRadius = l / 2.0 - r;
        double innerRadius = r0 + r;
        if (innerRadius >= outerRadius) {
            throw new IllegalArgumentException("Obstacle is too large for the given domain and particle radius.");
        }
    }

    public int getN() {
        return n;
    }

    public double getL() {
        return l;
    }

    public double getR0() {
        return r0;
    }

    public double getR() {
        return r;
    }

    public double getM() {
        return m;
    }

    public double getK() {
        return k;
    }

    public double getV0() {
        return v0;
    }

    public double getTf() {
        return tf;
    }

    public double getDt() {
        return dt;
    }

    public double getDt2() {
        return dt2;
    }

    public long getSeed() {
        return seed;
    }

    public String getOutPath() {
        return outPath;
    }

    public String getEventsPath() {
        return eventsPath;
    }

    public static void printUsageAndExit() {
        System.out.println("Usage: java ScanningRateSimulation [options]");
        System.out.println("Options:");
        System.out.println("  --n <value>          Number of particles");
        System.out.println("  --l <value>          Domain diameter (m)");
        System.out.println("  --r0 <value>         Obstacle radius (m)");
        System.out.println("  --r <value>          Particle radius (m)");
        System.out.println("  --m <value>          Particle mass (kg)");
        System.out.println("  --k <value>          Elastic constant (N/m)");
        System.out.println("  --v0 <value>         Initial speed (m/s)");
        System.out.println("  --tf <value>         Final time (s)");
        System.out.println("  --dt <value>         Integration step (s)");
        System.out.println("  --dt2 <value>        Output step (s)");
        System.out.println("  --seed <value>       Random seed");
        System.out.println("  --out <path>         Output state file path");
        System.out.println("  --events-out <path>  Output events file path");
        System.exit(0);
    }
}
