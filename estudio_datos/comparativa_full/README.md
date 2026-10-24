# Comparativa local de detección de fraude

Esta carpeta contiene la comparación ejecutada de 15 familias/variantes sobre los datos del taller, con 24 combinaciones de familia y régimen. Las 1.542 evaluaciones previstas han pasado la auditoría completa y una comprobación independiente de AP desde los scores.

Los comandos siguientes se ejecutan desde la **raíz del proyecto**, donde están los ZIP originales y `.venv`. Los notebooks canónicos son [02_comparativa_modelos.ipynb](../../02_comparativa_modelos.ipynb) y [03_modelo_final_y_entrega.ipynb](../../03_modelo_final_y_entrega.ipynb). El notebook original del taller y el diagnóstico previo se conservan como documentos separados.

Las métricas y salidas de entrenamiento conservadas corresponden a la ejecución original, anterior a las correcciones de reanudación y vecinos. Estas correcciones se validan por separado; no se han vuelto a entrenar los 1.542 ajustes ni se han reescrito sus huellas para atribuirlos al código nuevo. Los artefactos históricos permanecen en el checkout original y no se copian a este worktree. Una ejecución completa aquí genera sus propios resultados con las nuevas identidades de código. Solo las presentaciones de tiempos modificadas del notebook 02 se han vuelto a ejecutar con los registros históricos; su alcance consta en los metadatos de esas celdas.

## Resultado de la comparación

La selección interior sobre construcción completa eligió **LightGBM supervisado**, configuración 3: 500 árboles, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=50`, `reg_lambda=1` y ponderación `balanced`. Su AP media interior fue **0,847203**.

| Familia y régimen | AP exterior media | Desviación entre los cinco folds |
| --- | ---: | ---: |
| LightGBM supervisado | 0,841364 | 0,044755 |
| Random Forest supervisado | 0,834397 | 0,029824 |
| Logística supervisada | 0,760769 | 0,048248 |
| Autoencoder con normales | 0,657655 | 0,131401 |
| GMM fijo con normales | 0,648035 | 0,085213 |

El procedimiento anidado también escogió LightGBM en los cinco entrenamientos exteriores y alcanzó AP **0,841364**, con mejora pareada media de **0,193329** frente al GMM fijo. Al revisar aproximadamente el 1 % de operaciones de cada fold recuperó el **88,47 %** de los fraudes de validación. La diferencia media frente a Random Forest es pequeña; estos resultados no demuestran superioridad universal ni significación estadística.

Las fuentes completas son [tabla_resumen_familia_regimen.csv](evidencia/tabla_resumen_familia_regimen.csv), [seleccion_final.json](evidencia/seleccion_final.json) y [auditoria_independiente_cv.json](evidencia/auditoria_independiente_cv.json). La AP se estimó en validación interna, no en producción. Los 321 ajustes GMM declararon convergencia; 75 de las 82 búsquedas internas del AE alcanzaron el presupuesto de 200 épocas. Ese límite queda explícito y no se interpreta como convergencia del AE.

Los **24 refits finales están completos**, con sus modelos, scores y metadatos guardados. El LightGBM elegido se reajustó con las **170.884 filas** de construcción. La [entrega](../../entrega_scores.csv) contiene **113.923 scores**, verificados en el orden original. Ambos notebooks conservan todas sus salidas: 17 celdas de código en el 02 y 9 en el 03, sin errores. Cinco inferencias completas del modelo recargado coincidieron con el CSV; la mediana local fue **1,324 segundos** y el IQR **0,017 segundos**, excluyendo la carga del modelo. Véanse [auditoria_entrega.json](evidencia/auditoria_entrega.json) y [tiempos_inferencia_entrega.json](evidencia/tiempos_inferencia_entrega.json).

## Entorno

### Archivos publicados y resultados locales

Git conserva los notebooks ejecutados, el código, la entrega y una [selección de evidencia histórica](evidencia/README.md). Los enlaces a tablas y auditorías de este documento apuntan a esas copias en `evidencia/`. La tabla de rutas más abajo describe el directorio de ejecución `resultados/`, que se regenera y no se incluye en Git.

En una copia nueva, ejecutar primero el notebook 02 y después el 03. No copiar `evidencia/` a `resultados/`: la evidencia no contiene los modelos, arrays ni scores intermedios necesarios para reanudar. Las salidas de los notebooks conservan las rutas de su ejecución original.

El protocolo registrado utiliza **CPython 3.12.14**, CPU y las versiones fijadas en [requirements.txt](requirements.txt), obtenido del entorno local. Para reproducirlo en una copia nueva, con Python 3.12.14 disponible como `python3.12`:

```bash
env -u PYTHONPATH python3.12 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install -r estudio_datos/comparativa_full/requirements.txt
env -u PYTHONPATH .venv/bin/python -m ipykernel install --prefix .venv --name taller-comparativa --display-name "Python (taller-comparativa)"
```

Si `.venv` ya contiene el entorno del experimento, se reutiliza; no es necesario recrearlo ni reinstalar dependencias durante un entrenamiento. El kernel `taller-comparativa` debe apuntar a ese entorno. Se elimina `PYTHONPATH` de las llamadas para evitar importar paquetes de otro entorno.

El archivo de dependencias conserva el entorno de este Mac. Otro sistema operativo puede necesitar sus propios paquetes de plataforma; cambiar versiones constituye otro entorno y no permite asumir equivalencia numérica con el protocolo guardado. El protocolo registra las versiones principales y la versión exacta de Python.

## Ejecución y reanudación

Para ejecutar ambos notebooks, guardar sus salidas en los propios `.ipynb` y detenerse si una celda falla:

```bash
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.ejecutar_notebooks
```

El runner [ejecutar_notebooks.py](ejecutar_notebooks.py) usa el kernel `taller-comparativa`, trabaja desde la raíz, guarda después de cada celda ejecutada y registra inicio, fin y estado en `metadata.ejecucion_local`.

- **02** prepara y verifica los datos, ejecuta o reanuda toda la comparación con cuatro trabajos y dos hilos por trabajo, audita la cobertura completa y produce tablas, selección y figuras.
- **03** necesita los artefactos del 02. Reajusta las 24 combinaciones mediante hasta cuatro procesos independientes y dos hilos por modelo, verifica la entrega ganadora y, cuando terminan todos los refits, ejecuta cinco inferencias completas del modelo guardado, sin reentrenarlo, para registrar mediana e IQR del tiempo de inferencia.

También se puede ejecutar un notebook concreto:

```bash
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.ejecutar_notebooks 02_comparativa_modelos.ipynb
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.ejecutar_notebooks 03_modelo_final_y_entrega.ipynb
```

Para iniciar el entrenamiento directamente desde terminal:

```bash
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.ejecutar --todos --trabajos 4 --hilos 2
```

Ese comando prepara y sella los datos antes de lanzar los trabajos. Una ejecución posterior reutiliza los ajustes completos cuya identidad y hash siguen siendo válidos. El notebook 02 también llama a este comando: **no se debe lanzar mientras siga activo otro controlador del mismo experimento**, porque no existe un bloqueo automático contra ejecuciones simultáneas sobre los mismos resultados.

Los registros de cada trabajo están en `resultados/logs/`, el estado del controlador en `resultados/estado.json` y los fallos capturados en `resultados/errores/`. La mera existencia de resultados parciales no acredita que la comparación haya terminado.

Para completar una familia costosa repartiendo sus seis búsquedas entre procesos, una vez detenido cualquier otro entrenamiento de esa misma familia:

```bash
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.ejecutar_particiones --familia random_forest --regimen supervisado --trabajos 5 --hilos 2
```

Este lanzador usa las funciones originales de ajuste y selección, conserva los resultados válidos y verifica al final las decisiones con el recorrido original. Su estado está en `resultados/estado_particiones/`. En esta ejecución se conservó el primer fold completo de Random Forest y se reanudaron las búsquedas restantes en paralelo al terminar las otras familias; el cambio de planificación se registra en `resultados/logs/reprogramacion_random_forest.json`. Se amplió la concurrencia de cuatro a cinco procesos para repartir las cinco búsquedas pendientes y ocupar diez hilos de los once núcleos disponibles; el cierre cooperativo conservó los ajustes en curso. Esa reanudación figura en `resultados/logs/ajuste_concurrencia_random_forest.json`. No se redujeron datos, árboles ni configuraciones.

El ejecutor normaliza las referencias de metadatos de LOF/kNN desde los JSON persistidos: los diagnósticos de empates de la geometría compartida no deben modificar retrospectivamente un registro ya guardado. Esta normalización no cambia los scores.

La secuencia equivalente por módulos, después de terminar correctamente el entrenamiento, es:

```bash
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.analizar
env -u PYTHONPATH .venv/bin/python -m estudio_datos.comparativa_full.finalizar --hilos 2 --trabajos 4
env -u PYTHONPATH .venv/bin/python -c 'from estudio_datos.comparativa_full.finalizar import entregar; entregar()'
```

El CLI `finalizar` realiza los reajustes; la llamada posterior a `entregar()` verifica y copia la entrega. Esta ruta por módulos no genera las salidas y figuras de los notebooks ni la medición adicional de cinco inferencias. Para conservar la evidencia completa en los notebooks se utiliza el runner.

## Datos, particiones y selección

Los archivos de entrada son `credit_card_construction.csv.zip` y `credit_card_production.csv.zip`. Construcción contiene **170.884 filas, 295 fraudes y 170.589 normales**. Producción contiene **113.923 filas** y su columna `Class` carece de etiquetas. Se usan los 21 predictores `X1` a `X20` y `Amount`; `Class` nunca entra como predictor.

La validación es **anidada: cinco folds exteriores y tres interiores**, con `StratifiedGroupKFold` y semilla base `20261020`. Los grupos reúnen filas con valores exactamente iguales en los 21 predictores. Un grupo no cruza entre entrenamiento y evaluación. Esto controla la presencia de patrones idénticos en ambas particiones; no demuestra que esas filas sean la misma transacción ni agrupa observaciones solo parecidas.

Para cada entrenamiento exterior:

1. Los tres folds interiores eligen la configuración de cada familia/régimen.
2. Esa configuración se ajusta sobre todo el entrenamiento exterior permitido por el régimen y se evalúa en su fold exterior.
3. Para evaluar el procedimiento completo de selección, también se elige la familia/régimen usando únicamente sus métricas interiores y se consulta su evaluación exterior.

Una búsqueda adicional con tres folds sobre toda construcción, identificada como `outer=-1`, decide las configuraciones de los refits y el modelo final. Las métricas exteriores no intervienen en esa elección. Después, cada combinación se reajusta sobre toda construcción o sobre todas sus filas normales, según el régimen. Producción se puntúa después de seleccionar.

La métrica principal es **Average Precision (AP)** mediante `average_precision_score`, coherente con la función indicada en el taller. No se sustituye por el área trapezoidal bajo la curva precision–recall. Los scores siempre crecen con la sospecha de fraude.

La selección maximiza la AP media interior. Solo ante igualdad exacta se usan, en orden, mayor recall al 1 %, menor tiempo medio de ajuste y desempates deterministas por familia, régimen y configuración. Se registran además ROC AUC y precision/recall para presupuestos de revisión del 0,2 %, 1 % y 2 %. Los empates de score en esos presupuestos se resuelven por índice de fila y se contabilizan.

La AP exterior se resume por fold, con media, desviación típica y rango. No se calcula AP concatenando scores crudos de modelos ajustados en folds distintos. La desviación entre cinco folds es descriptiva, no un intervalo de confianza.

## Familias, regímenes y alcance del cómputo

| Familia en código | Configuraciones por régimen | Regímenes |
| --- | ---: | --- |
| `constante` | 1 | referencia |
| `univariante` | 1 | supervisado |
| `gmm_fijo` | 1 | normales |
| `gmm` | 8 | normales, todos |
| `isolation_forest` | 3 | normales, todos |
| `pca` | 5 | normales, todos |
| `lof` | 3 | normales, todos |
| `knn` | 3 | normales, todos |
| `autoencoder` | 2 | normales, todos |
| `rbig` | 2 | normales, todos |
| `ecod_frozen` | 1 | normales, todos |
| `ecod_cohorte` | 1 | normales, todos |
| `logistica` | 8 | supervisado |
| `random_forest` | 6 | supervisado |
| `lightgbm` | 6 | supervisado |

Son **24 combinaciones familia/régimen y 79 configuraciones contando regímenes**. La cobertura exigida es:

- `79 × 3 × (5 + 1) = 1.422` ajustes interiores: cinco búsquedas exteriores más la búsqueda final.
- `24 × 5 = 120` ajustes con evaluación exterior.
- **1.542 ajustes evaluados y registrados**, más **24 refits finales**.

Estos son recuentos lógicos del protocolo. LOF y kNN comparten una búsqueda de vecinos, y cada ajuste del autoencoder incluye selección interna de épocas y un nuevo entrenamiento. Por ello, el recuento no equivale a 1.542 llamadas independientes de idéntico coste.

En `normales`, el ajuste recibe solo las filas `Class=0` de su partición; esta exclusión usa etiquetas. En `todos`, el detector recibe todas las filas y no usa sus etiquetas durante el ajuste. Los clasificadores y el selector univariante utilizan ambas clases. Incluso los detectores sin etiquetas en `fit` tienen selección de hiperparámetros mediante AP etiquetada: el procedimiento completo no es una evaluación enteramente no supervisada.

`StandardScaler` se ajusta exclusivamente con las filas de entrenamiento del régimen. Se omite para constante, univariante, árboles y ECOD. No se recortan filas para acelerar la comparación; el muestreo interno por árbol de Isolation Forest forma parte de sus configuraciones explícitas.

## Adaptaciones que afectan a la interpretación

- **ECOD original (`ecod_cohorte`)** usa la implementación instalada de PyOD. Al consultar, sus distribuciones dependen conjuntamente de entrenamiento y cohorte consultada. Cada evaluación se puntúa como un lote completo. Se compara y reajusta por separado, pero **no participa en la selección principal**.
- **FrozenECOD (`ecod_frozen`)** es una adaptación: congela las distribuciones empíricas y el signo de asimetría del entrenamiento, suaviza los extremos y produce scores independientes del lote consultado. Sí participa en la selección principal. No se presenta como numéricamente equivalente al ECOD original.
- **RBIG** adapta el material docente con histogramas y pseudocuentas normalizados, colas exponenciales de masa explícita, Gaussianización y rotaciones PCA. Calcula directamente la logdensidad con sus jacobianos en `float64`; el score es su negativo. La profundidad se elige entre 20 y 50 capas, sin afirmar convergencia por entropía. Las distribuciones y rotaciones permanecen congeladas al predecir.
- **Autoencoder** adapta la reconstrucción a 21 variables estandarizadas, salida lineal y MSE por fila. Selecciona épocas mediante una reserva interna de aproximadamente el 10 % de grupos, considera épocas desde la 20 y después reajusta desde cero con todas las filas recibidas. Utiliza CPU y cálculo interno `float32`; el vector de salida se devuelve como `float64`.
- **LOF/kNN** usan vecinos euclídeos exactos en `float64`, buscando hasta 60 vecinos y reutilizando prefijos para cada `k`. LOF funciona en modo de novedad; kNN usa la distancia media a sus vecinos. Ante distancias empatadas, la elección del prefijo puede diferir de una búsqueda independiente para cada `k`. Sus costes compartidos no se deben sumar como trabajos independientes.

En los ajustes de validación con geometría compartida de LOF/kNN, `fit_s` incluye el escalado y las búsquedas de vecinos de **entrenamiento y consulta**. `score_s` mide solo la agregación del score a partir de esos vecinos; no representa por sí solo el coste de inferencia sobre nuevas filas. El notebook 02 muestra el desglose para los diez ajustes exteriores de LOF, sin duplicarlo en kNN. También admite ajustes individuales y mezclas de ambas rutas: en los individuales, `fit_s` y `score_s` son los tiempos completos de ajuste y consulta, y los desgloses no registrados aparecen como `NaN`. Los tiempos de los refits finales y las cinco inferencias del notebook 03 se registran por separado.

Las implementaciones, parámetros y referencias están en [modelos.py](modelos.py), [modelo_ecod.py](modelo_ecod.py), [modelo_rbig.py](modelo_rbig.py), [modelo_ae.py](modelo_ae.py) y [modelos_vecinos.py](modelos_vecinos.py).

## Integridad, cachés y artefactos

[experimento.py](experimento.py) registra hashes de los ZIP originales, arrays preparados, particiones y código relevante. Cada ajuste identifica familia, régimen, parámetros, semilla, índices de entrenamiento/evaluación y protocolo. La caché exige identidad coincidente, estado completo y hash válido del archivo de scores. Los cambios de protocolo no se aceptan silenciosamente sobre los resultados existentes; las versiones nuevas deben conservarse como otro experimento.

No se deben editar los módulos de entrenamiento mientras haya procesos activos: se comprueba que el código relevante no haya cambiado desde su importación. `cargar()` rechaza versiones de Python o librerías distintas del protocolo antes de leer los arrays, consultar cachés o entrenar; la comprobación cubre los lanzadores individuales, por particiones y de refit. El finalizador también contrasta decisiones auditadas e integridad de los artefactos. Se fijan semillas y controles de hilos; no se promete identidad bit a bit entre arquitecturas, librerías o equipos distintos. El tiempo usado para desempatar métricas exactamente iguales también puede variar entre ejecuciones.

Las regresiones se comprueban con datos sintéticos y archivos temporales, sin modificar los resultados históricos:

```bash
env -u PYTHONPATH .venv/bin/python -m unittest discover -s estudio_datos/comparativa_full/tests -v
```

| Ruta relativa a esta carpeta | Contenido y alcance |
| --- | --- |
| `resultados/protocolo.json` | Datos, versiones, rejilla, regímenes, semillas y reglas. |
| `resultados/integridad_datos.json` | Sello de arrays y particiones verificados frente a las fuentes. |
| `resultados/particiones.npz` y `resultados/particiones.csv` | Índices de las particiones y asignación exterior de cada fila. |
| `resultados/ajustes/` | JSON de identidad, métricas, tiempos y avisos; NPZ con fila, clase, grupo y score. |
| `resultados/decisiones/` | Selecciones interiores de cada familia/régimen para los cinco folds y `outer=-1`. |
| `resultados/tabla_por_fold.csv` | Evaluaciones exteriores de las 24 combinaciones. |
| `resultados/tabla_resumen_familia_regimen.csv` | Resumen descriptivo por familia/régimen. |
| `resultados/tabla_candidatos_interiores.csv` | Candidatos y criterios interiores usados para seleccionar. |
| `resultados/tabla_procedimiento_anidado.csv` | Evaluación exterior del selector completo. |
| `resultados/scores_oof/` | Scores fuera de entrenamiento, con fold y configuración; no son scores de un único modelo. |
| `resultados/seleccion_final.json` | Elección y fuentes auditadas anteriores al refit. |
| `resultados/auditoria_completitud.json` | Acreditación de los 1.542 ajustes; no acredita todavía producción. |
| `resultados/finales/` y `resultados/resumen_refits.json` | Modelo `.joblib`, CSV y metadata de cada uno de los 24 refits. |
| `resultados/auditoria_entrega.json` y `estado_entrega.json` | Verificación posterior de refits y entrega elegida. |
| `resultados/tiempos_inferencia_entrega.json` | Cinco inferencias completas, mediana/IQR y hashes, generados por el notebook 03. |

La entrega es [`entrega_scores.csv`](../../entrega_scores.csv), en la raíz: **una columna `score`, sin índice y con 113.923 filas de datos**, en el orden original de producción. Se copia del CSV de la combinación elegida después de verificar los 24 refits. Un score mayor significa más sospecha; no implica una probabilidad calibrada.

La verificación de serialización y orden de lote utiliza `rtol=1e-6`, `atol=1e-8`; kNN usa `atol=1e-7`. En una de las 258 consultas comprobadas, el cálculo euclídeo por productos escalares varió `3,82e-8` al invertir el lote. La comprobación por resta directa confirmó redondeo, con vecinos y ranking idénticos en esa muestra. Se conservan el [diagnóstico numérico](evidencia/logs/diagnostico_precision_knn.json), la incidencia inicial y las tolerancias y diferencias de cada refit. No se alteraron el algoritmo ni los scores de validación. La invariancia numérica no significa identidad bit a bit.

La auditoría de validación y la de entrega son documentos separados. `seleccion_final.json` conserva la decisión previa al refit; la finalización se acredita en `estado_entrega.json` y `auditoria_entrega.json`. Los CSV, modelos y auditorías solo existen cuando termina su fase correspondiente.

## Límites del estudio

Los datos de construcción ya fueron examinados durante el diagnóstico: esta es una **validación interna exploratoria**, no un test externo intacto. Los grupos controlan coincidencias exactas de predictores, pero no sustituyen una validación temporal o por cliente. La agrupación estratificada no garantiza idéntica prevalencia en todos los folds.

La escasez de fraudes limita la precisión de las estimaciones. La búsqueda selecciona entre la rejilla declarada; no demuestra optimalidad entre todos los algoritmos posibles. La selección usa etiquetas disponibles y las adaptaciones tienen los límites descritos arriba.

Producción no tiene etiquetas: permite generar scores y medir inferencia, pero **no calcular AP ni confirmar rendimiento futuro**. Las cinco mediciones del notebook 03 corresponden a la carga real de este equipo; excluyen la carga inicial del modelo y no certifican latencia de un servicio desplegado.
