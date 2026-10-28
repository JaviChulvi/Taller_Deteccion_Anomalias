# Aportación de Javi: detección de anomalías

Estudio de detección de fraude con los datos del taller. Esta carpeta conserva mis notebooks ejecutados, el código de los modelos y la entrega de scores. Los materiales originales y los dos ZIP de datos se comparten desde la [raíz del repositorio](../README.md).

## Recorrido del estudio

1. [Diagnóstico inicial](01_diagnostico_fraude.ipynb): datos, duplicados, señales y referencia GMM.
2. [Comparativa de modelos](02_comparativa_modelos.ipynb): 15 familias/variantes, 24 combinaciones de familia y régimen y 1.542 ajustes con validación anidada por grupos de predictores idénticos.
3. [Modelo final y entrega](03_modelo_final_y_entrega.ipynb): selección interior, 24 reajustes finales y comprobaciones de producción.

La selección eligió LightGBM supervisado. El procedimiento obtuvo AP exterior media **0,8414**, con desviación entre cinco folds **0,0448**. Es validación interna exploratoria; producción carece de etiquetas y su rendimiento es desconocido.

La [entrega de scores](entrega_scores.csv) contiene una columna `score` y 113.923 filas en el orden original de producción. Un valor mayor indica más sospecha, sin garantizar calibración probabilística.

## Reproducción y evidencia

Ejecutar los comandos desde `javi/` (`cd javi` desde la raíz) y abrir los notebooks con esa carpeta como directorio de trabajo. El entorno propio se crea en `javi/.venv/`; los scripts leen los ZIP desde la carpeta superior y escriben los resultados dentro de `javi/`.

El [README de la comparativa](estudio_datos/comparativa_full/README.md) detalla el entorno fijado, los comandos, los modelos y los límites. Los notebooks necesitan los módulos Python del repositorio y ambos ZIP originales; el notebook 03 necesita los resultados generados por el 02. Las salidas guardadas permiten leer el estudio sin entrenar.

La [evidencia histórica](estudio_datos/comparativa_full/evidencia/README.md) conserva tablas, decisiones, auditorías y figuras de la ejecución original. Los modelos serializados, arrays, scores intermedios y logs de ejecución se generan en `estudio_datos/comparativa_full/resultados/`, que está excluido de Git. Una copia nueva debe ejecutar el notebook 02 antes del 03; la evidencia publicada no es una caché de entrenamiento completa.

Las salidas guardadas conservan las rutas y hashes de la ejecución anterior a esta reorganización. No se han reentrenado los modelos ni modificado sus métricas. Los resultados locales antiguos no se trasladan automáticamente: no deben reutilizarse como caché de una ejecución con código modificado.

## Estado de la entrega académica

El trabajo computacional y el CSV están completos. Antes de entregar falta incorporar los nombres del grupo y reunir el contenido exigido por el enunciado en un notebook de entrega, comprobando su ejecución con los archivos que lo acompañen. Los notebooks de estudio y la publicación del repositorio no acreditan una entrega en el aula virtual.
