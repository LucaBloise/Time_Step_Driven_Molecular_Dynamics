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

        BufferedWriter writer = null;
        BufferedWriter events = null;

        if (!config.isNoState()) {
            OutputWriter.ensureParentDir(config.getOutPath());
            writer = Files.newBufferedWriter(Paths.get(config.getOutPath()), StandardCharsets.US_ASCII);
            OutputWriter.writeHeader(writer, config);
        }

        if (!config.isNoEvents()) {
            OutputWriter.ensureParentDir(config.getEventsPath());
            events = Files.newBufferedWriter(Paths.get(config.getEventsPath()), StandardCharsets.US_ASCII);
            OutputWriter.writeEventsHeader(events, config);
        }

        try {
            long executionTimeMs = SimulationRunner.run(config, state, writer, events);

            if (config.getPropertiesPath() != null && !config.getPropertiesPath().isBlank()) {
                OutputWriter.writeMetadata(config.getPropertiesPath(), config, executionTimeMs);
            }
        } finally {
            if (writer != null) {
                writer.close();
            }
            if (events != null) {
                events.close();
            }
        }
    }
}
