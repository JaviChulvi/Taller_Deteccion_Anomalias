# Taller de detección de anomalías

Estudio de detección de fraude con los datos del taller. El repositorio conserva los materiales originales, los notebooks ejecutados, el código de los modelos y la entrega de scores.

## Recorrido del estudio

1. [Diagnóstico inicial](01_diagnostico_fraude.ipynb): datos, duplicados, señales y referencia GMM.
2. [Comparativa de modelos](02_comparativa_modelos.ipynb): 15 familias/variantes, 24 combinaciones de familia y régimen y 1.542 ajustes con validación anidada por grupos de predictores idénticos.
3. [Modelo final y entrega](03_modelo_final_y_entrega.ipynb): selección interior, 24 reajustes finales y comprobaciones de producción.

La selección eligió LightGBM supervisado. El procedimiento obtuvo AP exterior media **0,8414**, con desviación entre cinco folds **0,0448**. Es validación interna exploratoria; producción carece de etiquetas y su rendimiento es desconocido.

La [entrega de scores](entrega_scores.csv) contiene una columna `score` y 113.923 filas en el orden original de producción. Un valor mayor indica más sospecha, sin garantizar calibración probabilística.

## Reproducción y evidencia

El [README de la comparativa](estudio_datos/comparativa_full/README.md) detalla el entorno fijado, los comandos, los modelos y los límites. Los notebooks necesitan los módulos Python del repositorio y ambos ZIP originales; el notebook 03 necesita los resultados generados por el 02. Las salidas guardadas permiten leer el estudio sin entrenar.

La [evidencia histórica](estudio_datos/comparativa_full/evidencia/README.md) conserva tablas, decisiones, auditorías y figuras de la ejecución original. Los modelos serializados, arrays, scores intermedios y logs de ejecución se generan en `estudio_datos/comparativa_full/resultados/`, que está excluido de Git. Una copia nueva debe ejecutar el notebook 02 antes del 03; la evidencia publicada no es una caché de entrenamiento completa.

## Estado de la entrega académica

El trabajo computacional y el CSV están completos. Antes de entregar falta incorporar los nombres del grupo y reunir el contenido exigido por el enunciado en un notebook de entrega, comprobando su ejecución con los archivos que lo acompañen. Los notebooks de estudio y la publicación del repositorio no acreditan una entrega en el aula virtual.
