# Visualization scripts (quick run)

Ejecutar desde la raiz del repo.

## Requisitos

- Python 3.10+.
- Dependencias: `pip install matplotlib numpy`.
- Para scripts que lanzan simulaciones TP4: Java disponible (si hace falta, pasar `--java-cmd "ruta/al/java"`).
- Para comparaciones TP3 vs TP4: tener tambien el repo TP3.

## Scripts y parametros clave

1. `runtime_vs_n.py`
- Que hace: corre benchmarks de TP4 y grafica tiempo de ejecucion vs N.
- Comando minimo: `python visualization/runtime_vs_n.py`
- Parametros utiles: `--n-values`, `--repetitions`, `--tf`, `--out-csv`, `--out-figure`, `--tp3-csv`, `--log-y`.
- Ejemplo: `python visualization/runtime_vs_n.py --n-values 100,200,300 --repetitions 3 --tf 500 --out-figure outputs/runtime/runtime_vs_n_demo.png`

2. `scanning_rate_vs_n.py`
- Que hace: calcula y grafica tasa de escaneo J vs N (incluye figuras diagnosticas Cfc).
- Comando minimo: `python visualization/scanning_rate_vs_n.py`
- Parametros utiles: `--n-values`, `--repetitions`, `--tf`, `--dt`, `--out-csv`, `--out-figure`, `--tp3-csv`, `--cfc-low-n`, `--cfc-high-n`.
- Ejemplo: `python visualization/scanning_rate_vs_n.py --n-values 100,200,300 --repetitions 3 --tf 800 --dt 0.001 --out-figure outputs/scanning_rate_tp4/scanning_demo.png`

3. `radial_profiles.py`
- Que hace: genera perfiles radiales (rho, |v|, Jin) y analisis cerca del obstaculo.
- Comando minimo: `python visualization/radial_profiles.py`
- Parametros utiles: `--n-values`, `--repetitions`, `--tf`, `--dt`, `--dt2`, `--ds`, `--stationary-start`, `--out-csv`, `--only-plot`, `--reuse-existing-runs`.
- Ejemplo: `python visualization/radial_profiles.py --n-values 100,200,300 --repetitions 2 --tf 600 --dt 0.001 --dt2 0.1 --ds 0.2`

4. `k_variation_analysis.py`
- Que hace: analiza el efecto de variar k en J, Jin y escalares derivados.
- Comando minimo: `python visualization/k_variation_analysis.py`
- Parametros utiles: `--k-values`, `--n-values`, `--repetitions`, `--tf`, `--dt`, `--dt2`, `--target-s`, `--out-csv`, `--only-plot`, `--reuse-existing-runs`.
- Ejemplo: `python visualization/k_variation_analysis.py --k-values 1e2,1e3,1e4 --n-values 100,200,300 --repetitions 2 --target-s 2.1`

5. `first_used_return_time.py`
- Que hace: calcula el tiempo de retorno de la primera particula usada.
- Comando minimo: `python visualization/first_used_return_time.py`
- Parametros utiles: `--k-values`, `--n-values`, `--repetitions`, `--tf`, `--dt`, `--dt2`, `--out-csv`, `--figure-n`, `--figure-k`, `--only-plot`.
- Ejemplo: `python visualization/first_used_return_time.py --k-values 1e2,1e3 --n-values 100,200,300 --repetitions 3 --tf 1000`

6. `plot_scanning_rate_energy.py`
- Que hace: grafica energia total (o relativa) vs tiempo desde un `states.txt` de scanning.
- Comando minimo: `python visualization/plot_scanning_rate_energy.py`
- Parametros utiles: `--input` o `--input-dir`, `--out`, `--relative`, `--y-min`, `--y-max`, `--y-pad`.
- Ejemplo: `python visualization/plot_scanning_rate_energy.py --input-dir outputs/scanningRate --out outputs/scanningRate/energy_relative.png --relative`

7. `plot_scanning_rate_animation.py`
- Que hace: crea animacion MP4/GIF de las particulas frescas/usadas.
- Comando minimo: `python visualization/plot_scanning_rate_animation.py`
- Parametros utiles: `--input` o `--input-dir`, `--out`, `--fps`, `--stride`, `--max-frames`, `--t-start`, `--t-end`.
- Ejemplo: `python visualization/plot_scanning_rate_animation.py --input-dir outputs/scanningRate --out outputs/scanningRate/anim.gif --fps 20 --stride 5 --t-start 100 --t-end 300`

8. `plot_oscillator_comparison.py`
- Que hace: compara solucion analitica vs metodos numericos del oscilador (full y zoom).
- Comando minimo: `python visualization/plot_oscillator_comparison.py`
- Parametros utiles: `--input-dir`, `--dt`, `--zoom`, `--full`, `--out-prefix`, `--zoom-start`, `--zoom-end`, `--zoom-ymin`, `--zoom-ymax`.
- Ejemplo: `python visualization/plot_oscillator_comparison.py --input-dir outputs/oscillatorOutputs --dt 0.01 --full --zoom --out-prefix outputs/oscillatorOutputs/comp_dt001`

9. `plot_oscillator_ecm_vs_dt.py`
- Que hace: grafica ECM de posicion vs dt para los metodos del oscilador.
- Comando minimo: `python visualization/plot_oscillator_ecm_vs_dt.py`
- Parametros utiles: `--input-dir`, `--out`, `--common-only`.
- Ejemplo: `python visualization/plot_oscillator_ecm_vs_dt.py --input-dir outputs/oscillatorOutputs --out outputs/oscillatorOutputs/ecm_vs_dt.png --common-only`

10. `compare_runtime_tp3_vs_tp4.py`
- Que hace: ejecuta/usa resultados de TP3 y TP4 para comparar runtime en una sola figura.
- Comando minimo: `python visualization/compare_runtime_tp3_vs_tp4.py`
- Parametros utiles: `--tp3-script`, `--tp3-out`, `--tp4-script`, `--tp4-out`, `--figure`, `--force-run`.
- Ejemplo: `python visualization/compare_runtime_tp3_vs_tp4.py --figure visualization/runtime_tp4_vs_tp3_demo.png --force-run`

11. `compare_scanning_tp3_vs_tp4.py`
- Que hace: ejecuta/usa resultados de TP3 y TP4 para comparar scanning rate y generar Cfc de TP4.
- Comando minimo: `python visualization/compare_scanning_tp3_vs_tp4.py`
- Parametros utiles: `--tp3-script`, `--tp3-out`, `--tp4-script`, `--tp4-out`, `--figure`, `--cfc-output-dir`, `--force-run`.
- Ejemplo: `python visualization/compare_scanning_tp3_vs_tp4.py --figure visualization/scanning_tp4_vs_tp3_demo.png --cfc-output-dir visualization --force-run`

## Tip rapido

- Ver todas las opciones: `python visualization/<script>.py --help`
