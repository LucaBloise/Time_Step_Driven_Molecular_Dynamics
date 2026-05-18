import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

public class OscillatorSimulation {
    private enum Method {
        EULER,
        VERLET,
        BEEMAN,
        GEAR5
    }

    private static final class Params {
        double m = 70.0;
        double k = 1.0e4;
        double gamma = 100.0;
        double tf = 5.0;
        double dt = 0.01;
        double dt2 = 0.01;
        double r0 = 1.0;
        double v0 = Double.NaN; // if NaN, compute from slide-36 condition
        Method method = Method.GEAR5;
        String outPath = null;
    }

    public static void main(String[] args) throws IOException {
        Params params = parseArgs(args);
        if (Double.isNaN(params.v0)) {
            params.v0 = -params.gamma * params.r0 / (2.0 * params.m);
        }
        if (params.outPath == null || params.outPath.isBlank()) {
            params.outPath = defaultOutputPath(params.method);
        }

        validate(params);
        ensureParentDir(params.outPath);

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(params.outPath), StandardCharsets.US_ASCII)) {
            writeHeader(writer, params);
            switch (params.method) {
                case EULER -> simulateEuler(params, writer);
                case VERLET -> simulateVerlet(params, writer);
                case BEEMAN -> simulateBeeman(params, writer);
                case GEAR5 -> simulateGear5(params, writer);
            }
        }
    }

    private static Params parseArgs(String[] args) {
        Params params = new Params();
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
                case "method" -> params.method = parseMethod(value);
                case "dt" -> params.dt = Double.parseDouble(value);
                case "dt2" -> params.dt2 = Double.parseDouble(value);
                case "tf" -> params.tf = Double.parseDouble(value);
                case "m" -> params.m = Double.parseDouble(value);
                case "k" -> params.k = Double.parseDouble(value);
                case "gamma" -> params.gamma = Double.parseDouble(value);
                case "r0" -> params.r0 = Double.parseDouble(value);
                case "v0" -> params.v0 = Double.parseDouble(value);
                case "out" -> params.outPath = value;
                default -> throw new IllegalArgumentException("Unknown option: --" + key);
            }
        }
        if (params.dt2 <= 0.0) {
            params.dt2 = params.dt;
        }
        return params;
    }

    private static void validate(Params params) {
        if (params.dt <= 0.0 || params.tf <= 0.0) {
            throw new IllegalArgumentException("dt and tf must be positive.");
        }
        if (params.m <= 0.0 || params.k <= 0.0) {
            throw new IllegalArgumentException("m and k must be positive.");
        }
    }

    private static void writeHeader(BufferedWriter writer, Params params) throws IOException {
        writer.write("# Damped oscillator (System 1)\n");
        writer.write(String.format(Locale.US, "# method=%s\n", params.method));
        writer.write(String.format(Locale.US, "# m=%.8f kg\n", params.m));
        writer.write(String.format(Locale.US, "# k=%.8f N/m\n", params.k));
        writer.write(String.format(Locale.US, "# gamma=%.8f kg/s\n", params.gamma));
        writer.write(String.format(Locale.US, "# r0=%.8f m\n", params.r0));
        writer.write(String.format(Locale.US, "# v0=%.8f m/s\n", params.v0));
        writer.write(String.format(Locale.US, "# dt=%.8f s\n", params.dt));
        writer.write(String.format(Locale.US, "# dt2=%.8f s\n", params.dt2));
        writer.write(String.format(Locale.US, "# tf=%.8f s\n", params.tf));
        writer.write("# columns: t r v a\n");
    }

    private static void simulateEuler(Params params, BufferedWriter writer) throws IOException {
        double t = 0.0;
        double r = params.r0;
        double v = params.v0;
        double a = accel(params, r, v);
        double dt = params.dt;

        double nextOutputTime = 0.0;
        int steps = (int) Math.ceil(params.tf / dt);

        for (int step = 0; step <= steps; step++) {
            if (t + 1e-12 >= nextOutputTime) {
                writeRow(writer, t, r, v, a);
                nextOutputTime += params.dt2;
            }
            if (step == steps) {
                break;
            }

            double rNext = r + v * dt;
            double vNext = v + a * dt;

            r = rNext;
            v = vNext;
            t += dt;
            a = accel(params, r, v);
        }
    }

    private static void simulateVerlet(Params params, BufferedWriter writer) throws IOException {
        double t = 0.0;
        double r = params.r0;
        double v = params.v0;
        double a = accel(params, r, v);

        double rPrev = r - v * params.dt + 0.5 * a * params.dt * params.dt;
        double nextOutputTime = 0.0;
        int steps = (int) Math.ceil(params.tf / params.dt);

        for (int step = 0; step <= steps; step++) {
            if (t + 1e-12 >= nextOutputTime) {
                writeRow(writer, t, r, v, a);
                nextOutputTime += params.dt2;
            }
            if (step == steps) {
                break;
            }

            double aNow = accel(params, r, v);
            double rNext = 2.0 * r - rPrev + aNow * params.dt * params.dt;
            double vNext = (rNext - rPrev) / (2.0 * params.dt);

            t += params.dt;
            rPrev = r;
            r = rNext;
            v = vNext;
            a = accel(params, r, v);
        }
    }

    private static void simulateBeeman(Params params, BufferedWriter writer) throws IOException {
        double t = 0.0;
        double r = params.r0;
        double v = params.v0;
        double a = accel(params, r, v);

        double rPrev = r - v * params.dt + 0.5 * a * params.dt * params.dt;
        double vPrev = v - a * params.dt;
        double aPrev = accel(params, rPrev, vPrev);

        double nextOutputTime = 0.0;
        int steps = (int) Math.ceil(params.tf / params.dt);

        for (int step = 0; step <= steps; step++) {
            if (t + 1e-12 >= nextOutputTime) {
                writeRow(writer, t, r, v, a);
                nextOutputTime += params.dt2;
            }
            if (step == steps) {
                break;
            }

            double rNext = r + v * params.dt + (2.0 / 3.0) * a * params.dt * params.dt
                    - (1.0 / 6.0) * aPrev * params.dt * params.dt;
            double vPred = v + 1.5 * a * params.dt - 0.5 * aPrev * params.dt;
            double aNext = accel(params, rNext, vPred);
            double vNext = v + (1.0 / 3.0) * aNext * params.dt + (5.0 / 6.0) * a * params.dt
                    - (1.0 / 6.0) * aPrev * params.dt;

            t += params.dt;
            rPrev = r;
            vPrev = v;
            aPrev = a;
            r = rNext;
            v = vNext;
            a = aNext;
        }
    }

    private static void simulateGear5(Params params, BufferedWriter writer) throws IOException {
        double t = 0.0;
        double r0 = params.r0;
        double r1 = params.v0;
        double r2 = accel(params, r0, r1);
        double r3 = accelDerivative(params, r1, r2);
        double r4 = accelDerivative(params, r2, r3);
        double r5 = accelDerivative(params, r3, r4);

        double nextOutputTime = 0.0;
        int steps = (int) Math.ceil(params.tf / params.dt);

        for (int step = 0; step <= steps; step++) {
            if (t + 1e-12 >= nextOutputTime) {
                writeRow(writer, t, r0, r1, r2);
                nextOutputTime += params.dt2;
            }
            if (step == steps) {
                break;
            }

            double dt = params.dt;
            double dt2 = dt * dt;
            double dt3 = dt2 * dt;
            double dt4 = dt3 * dt;
            double dt5 = dt4 * dt;

            double rp0 = r0 + r1 * dt + r2 * dt2 / 2.0 + r3 * dt3 / 6.0 + r4 * dt4 / 24.0 + r5 * dt5 / 120.0;
            double rp1 = r1 + r2 * dt + r3 * dt2 / 2.0 + r4 * dt3 / 6.0 + r5 * dt4 / 24.0;
            double rp2 = r2 + r3 * dt + r4 * dt2 / 2.0 + r5 * dt3 / 6.0;
            double rp3 = r3 + r4 * dt + r5 * dt2 / 2.0;
            double rp4 = r4 + r5 * dt;
            double rp5 = r5;

            double r2New = accel(params, rp0, rp1);
            double delta = r2New - rp2;

            double a0 = 3.0 / 16.0;
            double a1 = 251.0 / 360.0;
            double a2 = 1.0;
            double a3 = 11.0 / 18.0;
            double a4 = 1.0 / 6.0;
            double a5 = 1.0 / 60.0;

            r0 = rp0 + a0 * delta * dt2 / 2.0;
            r1 = rp1 + a1 * delta * dt;
            r2 = rp2 + a2 * delta;
            r3 = rp3 + a3 * delta / dt;
            r4 = rp4 + a4 * delta / dt2;
            r5 = rp5 + a5 * delta / dt3;

            t += dt;
        }
    }

    private static double accel(Params params, double r, double v) {
        return -(params.k / params.m) * r - (params.gamma / params.m) * v;
    }

    private static double accelDerivative(Params params, double r1, double r2) {
        return -(params.k / params.m) * r1 - (params.gamma / params.m) * r2;
    }

    private static void writeRow(BufferedWriter writer, double t, double r, double v, double a) throws IOException {
        writer.write(String.format(Locale.US, "%.8f %.10e %.10e %.10e\n", t, r, v, a));
    }

    private static Method parseMethod(String value) {
        return switch (value.toLowerCase(Locale.ROOT)) {
            case "euler" -> Method.EULER;
            case "verlet" -> Method.VERLET;
            case "beeman" -> Method.BEEMAN;
            case "gear5", "gear" -> Method.GEAR5;
            default -> throw new IllegalArgumentException("Unknown method: " + value);
        };
    }

    private static void ensureParentDir(String outPath) throws IOException {
        Path path = Paths.get(outPath);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    private static String defaultOutputPath(Method method) {
        String name = method.name().toLowerCase(Locale.ROOT);
        return "outputs/oscillatorOutputs/" + name + ".txt";
    }

    private static void printUsageAndExit() {
        System.out.println("Usage: java OscillatorSimulation --method <euler|verlet|beeman|gear5> [options]");
        System.out.println("Options:");
        System.out.println("  --dt <value>     Integration step (s)");
        System.out.println("  --dt2 <value>    Output step (s) (default: dt)");
        System.out.println("  --tf <value>     Final time (s)");
        System.out.println("  --m <value>      Mass (kg)");
        System.out.println("  --k <value>      Spring constant (N/m)");
        System.out.println("  --gamma <value>  Damping coefficient (kg/s)");
        System.out.println("  --r0 <value>     Initial position (m)");
        System.out.println("  --v0 <value>     Initial velocity (m/s)");
        System.out.println("  --out <path>     Output file path");
        System.exit(0);
    }
}
