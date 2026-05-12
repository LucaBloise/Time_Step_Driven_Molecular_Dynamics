import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ScanningRateSimulation {
    public static void main(String[] args) throws IOException {
        SimulationConfig config = SimulationConfig.fromArgs(args);
        config.applyDefaults();
        config.validate();

        SimulationState state = Initializer.initialize(config);

        OutputWriter.ensureParentDir(config.getOutPath());
        OutputWriter.ensureParentDir(config.getEventsPath());

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(config.getOutPath()), StandardCharsets.US_ASCII);
             BufferedWriter events = Files.newBufferedWriter(Paths.get(config.getEventsPath()), StandardCharsets.US_ASCII)) {
            OutputWriter.writeHeader(writer, config);
            OutputWriter.writeEventsHeader(events, config);
            SimulationRunner.run(config, state, writer, events);
        }
    }
}
