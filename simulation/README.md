# Simulation: guia completa

Este modulo contiene:

1. Simulacion de particulas en dominio circular: `ScanningRateSimulation`.
2. Simulacion de oscilador amortiguado: `OscillatorSimulation`.
3. Animacion de particulas via Python: `simulation/animate_particles.py`.

Todos los comandos deben ejecutarse desde la raiz del repo.

## Requisitos

1. Java 17+.
2. Python 3.10+.
3. Paquetes Python para animacion: `matplotlib`.
4. Opcional para MP4: `ffmpeg` en PATH.

## Compilar Java

```powershell
javac simulation/*.java
```

## ScanningRateSimulation (System 2)

Comando base:

```powershell
java -cp simulation ScanningRateSimulation [opciones]
```

Ejemplo:

```powershell
java -cp simulation ScanningRateSimulation --n 200 --k 1000 --tf 200 --dt 0.001 --dt2 0.1
```

Parametros mas usados:

1. `--n` cantidad de particulas.
2. `--l` diametro del dominio.
3. `--r0` radio del obstaculo central.
4. `--r` radio de particula.
5. `--m` masa.
6. `--k` constante elastica.
7. `--v0` velocidad inicial.
8. `--tf` tiempo final.
9. `--dt` paso de integracion.
10. `--dt2` paso de guardado.
11. `--seed` semilla.
12. `--out` archivo de estados.
13. `--events-out` archivo de eventos.
14. `--properties-out` archivo de metadata.
15. `--no-state`, `--no-events`, `--no-output` para desactivar salidas.

Salida por defecto: `outputs/scanningRate/`.

## OscillatorSimulation (System 1)

Comando base:

```powershell
java -cp simulation OscillatorSimulation --method <euler|verlet|beeman|gear5> [opciones]
```

Ejemplo:

```powershell
java -cp simulation OscillatorSimulation --method gear5 --dt 0.001 --dt2 0.001 --tf 5 --out outputs/oscillatorOutputs/gear5_dt1e-3.txt
```

Parametros mas usados:

1. `--method` metodo numerico: `euler`, `verlet`, `beeman`, `gear5`.
2. `--dt` paso de integracion.
3. `--dt2` paso de guardado (default: `dt`).
4. `--tf` tiempo final.
5. `--m`, `--k`, `--gamma` parametros fisicos.
6. `--r0`, `--v0` condiciones iniciales.
7. `--out` archivo de salida.

Salida por defecto: `outputs/oscillatorOutputs/<method>.txt`.

## Animate particles desde simulation

Se agrego el launcher `simulation/animate_particles.py` para correr la animacion desde esta carpeta de modulo.

Comando base:

```powershell
python simulation/animate_particles.py [opciones]
```

Modo 1: animar un archivo `states.txt` especifico.

```powershell
python simulation/animate_particles.py --states outputs/sweep_n_100_1000/n300_rep1/states.txt --out outputs/videos/anim_n300.gif
```

Modo 2: buscar corrida existente por N y tf.

```powershell
python simulation/animate_particles.py --n 300 --tf 2000 --use-existing --outputs-root outputs/sweep_n_100_1000 --out outputs/videos/anim_n300.mp4
```

Modo 3: generar corrida nueva y animar.

```powershell
python simulation/animate_particles.py --n 300 --tf 2000 --dt 0.001 --dt2 0.1 --k 1000 --out outputs/videos/anim_n300.gif
```

Opciones utiles de animacion:

1. `--fps` cuadros por segundo.
2. `--frame-step` toma 1 de cada N frames para acelerar.
3. `--max-frames` limita cantidad de frames.
4. `--dpi` resolucion final.
5. `--out` con extension `.gif` o `.mp4`.

Para MP4 asegurate de tener ffmpeg instalado.

## Ayuda

```powershell
java -cp simulation ScanningRateSimulation --help
java -cp simulation OscillatorSimulation --help
python simulation/animate_particles.py --help
```
