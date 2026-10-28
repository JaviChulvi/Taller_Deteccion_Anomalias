# Evidencia histórica de la comparativa

Estos archivos son copias sin modificaciones de una selección de `resultados/` de la ejecución local del 20 de octubre de 2026. El código y los notebooks originales están conservados en el commit `01f0d61`. Las auditorías describen esa ejecución; sus fechas, hashes, rutas locales y estados intermedios se conservan como procedencia, no como certificación de ejecuciones posteriores.

## Contenido publicado

- Protocolo, datos descriptivos, asignaciones exteriores, tablas comparativas y selección final.
- Las 24 decisiones por familia/régimen y los metadatos de los 24 reajustes finales.
- Auditorías de completitud, validación independiente y entrega, junto con las mediciones de inferencia.
- Figuras y registros JSON que explican las decisiones de ejecución y la precisión numérica de vecinos.
- `SHA256SUMS`, para comprobar que estas copias históricas no han cambiado.

La entrega seleccionada está en [entrega_scores.csv](../../../entrega_scores.csv). No se duplican los 24 CSV finales. Tampoco se publican los modelos `.joblib`, arrays `.npy`, particiones `.npz`, los 1.542 pares de metadatos/scores de ajuste, scores OOF ni logs de consola y estados de procesos. Estos artefactos regenerables permanecen en la copia local original; la carpeta completa ocupa aproximadamente 1,3 GB.

Por ello, los hashes de archivos omitidos son referencias de procedencia: **no permiten repetir la auditoría completa solo con esta carpeta**. Los notebooks conservan enlaces y rutas de su ejecución en `resultados/`; las copias publicadas están aquí. Los estados anteriores al refit en `seleccion_final.json` se complementan con `estado_entrega.json` y `auditoria_entrega.json`, que acreditan la finalización histórica.

## Comprobar las copias y el CSV

Desde esta carpeta, `shasum -a 256 -c SHA256SUMS` verifica todos los archivos de evidencia inventariados. Desde `javi/`, el siguiente comando comprueba el hash del CSV contra la auditoría de entrega, su forma y que todos los scores sean finitos, usando solo la biblioteca estándar:

```bash
python3 - <<'PY'
import csv
import hashlib
import json
import math
from pathlib import Path

raiz = Path('.')
auditoria = json.loads((raiz / 'estudio_datos/comparativa_full/evidencia/auditoria_entrega.json').read_text())
entrega = raiz / 'entrega_scores.csv'
assert hashlib.sha256(entrega.read_bytes()).hexdigest() == auditoria['scores_sha256']
with entrega.open(newline='') as archivo:
    lector = csv.reader(archivo)
    assert next(lector) == ['score']
    filas = 0
    for fila in lector:
        assert len(fila) == 1 and math.isfinite(float(fila[0]))
        filas += 1
assert filas == auditoria['filas'] == 113923
print('CSV verificado: 113.923 scores finitos y hash histórico coincidente.')
PY
```

Para una ejecución nueva, seguir los comandos del [README de la comparativa](../README.md). No copiar esta evidencia a `resultados/`: faltan los arrays y scores necesarios para reanudar. El ejecutor genera una carpeta `resultados/` independiente y mantiene intacta esta evidencia. Repetir el benchmark completo puede ser costoso; no es necesario para leer las salidas ya guardadas.
