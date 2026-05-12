import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

public class OutputWriter {
    public static void ensureParentDir(String outPath) throws IOException {
        Path path = Paths.get(outPath);
        Path parent = path.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }

    public static void writeHeader(BufferedWriter writer, SimulationConfig config) throws IOException {
        writer.write("# System 2: scanning rate in circular domain\n");
        writer.write(String.format(Locale.US, "# N=%d\n", config.getN()));
        writer.write(String.format(Locale.US, "# L=%.8f m\n", config.getL()));
        writer.write(String.format(Locale.US, "# r0=%.8f m\n", config.getR0()));
        writer.write(String.format(Locale.US, "# r=%.8f m\n", config.getR()));
        writer.write(String.format(Locale.US, "# m=%.8f kg\n", config.getM()));
        writer.write(String.format(Locale.US, "# k=%.8f N/m\n", config.getK()));
        writer.write(String.format(Locale.US, "# v0=%.8f m/s\n", config.getV0()));
        writer.write(String.format(Locale.US, "# tf=%.8f s\n", config.getTf()));
        writer.write(String.format(Locale.US, "# dt=%.8f s\n", config.getDt()));
        writer.write(String.format(Locale.US, "# dt2=%.8f s\n", config.getDt2()));
        writer.write(String.format(Locale.US, "# seed=%d\n", config.getSeed()));
        writer.write("# columns: t id x y vx vy state\n");
        writer.write("# state: 1=fresh, 0=used\n");
    }

    public static void writeEventsHeader(BufferedWriter writer, SimulationConfig config) throws IOException {
        writer.write("# System 2 events: fresh/used transitions\n");
        writer.write(String.format(Locale.US, "# N=%d\n", config.getN()));
        writer.write(String.format(Locale.US, "# L=%.8f m\n", config.getL()));
        writer.write(String.format(Locale.US, "# r0=%.8f m\n", config.getR0()));
        writer.write(String.format(Locale.US, "# r=%.8f m\n", config.getR()));
        writer.write(String.format(Locale.US, "# m=%.8f kg\n", config.getM()));
        writer.write(String.format(Locale.US, "# k=%.8f N/m\n", config.getK()));
        writer.write(String.format(Locale.US, "# v0=%.8f m/s\n", config.getV0()));
        writer.write(String.format(Locale.US, "# tf=%.8f s\n", config.getTf()));
        writer.write(String.format(Locale.US, "# dt=%.8f s\n", config.getDt()));
        writer.write(String.format(Locale.US, "# seed=%d\n", config.getSeed()));
        writer.write("# columns: t id event\n");
    }

    public static void writeState(BufferedWriter writer, SimulationState state, double time) throws IOException {
        StringBuilder builder = new StringBuilder();
        double[] x = state.getX();
        double[] y = state.getY();
        double[] vx = state.getVx();
        double[] vy = state.getVy();
        boolean[] isFresh = state.getIsFresh();

        for (int i = 0; i < state.getSize(); i++) {
            builder.append(String.format(Locale.US, "%.8f %d %.10e %.10e %.10e %.10e %d\n",
                    time,
                    i,
                    x[i],
                    y[i],
                    vx[i],
                    vy[i],
                    isFresh[i] ? 1 : 0));
        }
        writer.write(builder.toString());
    }
}
