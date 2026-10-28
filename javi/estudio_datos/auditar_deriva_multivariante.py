"""Audita el origen de las filas y la estructura numérica sin modelar fraude.

Ejecutar con un entorno que tenga numpy, pandas y scikit-learn:
    python javi/estudio_datos/auditar_deriva_multivariante.py

No modifica los ZIP, el notebook ni sus etiquetas. La etiqueta auxiliar representa
el archivo de origen y nunca se utiliza Class ni la posición de la fila.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def codificar_grupos(matriz: pd.DataFrame) -> np.ndarray:
    """Agrupa por igualdad de todos los valores, sin hashes susceptibles de colisión."""
    return pd.factorize(pd.MultiIndex.from_frame(matriz), sort=False)[0]


def bootstrap_auc_grupos(
    etiquetas: np.ndarray,
    probabilidades: np.ndarray,
    grupos: np.ndarray,
    repeticiones: int,
    semilla: int,
) -> dict:
    """Remuestrea grupos completos; conserva juntas las observaciones duplicadas.

    Mantiene fijo el modelo ajustado. El intervalo refleja la variación de la
    muestra de evaluación, no toda la variación por entrenamiento o partición.
    """
    rng = np.random.default_rng(semilla)
    orden = np.argsort(probabilidades, kind="stable")
    y = etiquetas[orden]
    p = probabilidades[orden]
    _, grupos_locales = np.unique(grupos[orden], return_inverse=True)
    n_grupos = int(grupos_locales.max()) + 1
    estratos_puntuacion = np.cumsum(np.r_[True, p[1:] != p[:-1]]) - 1
    n_puntuaciones = int(estratos_puntuacion.max()) + 1
    aucs = []
    for _ in range(repeticiones):
        multiplicidades = rng.multinomial(
            n_grupos, np.full(n_grupos, 1.0 / n_grupos)
        )
        pesos = multiplicidades[grupos_locales]
        positivos = np.bincount(
            estratos_puntuacion, weights=pesos * y, minlength=n_puntuaciones
        )
        negativos = np.bincount(
            estratos_puntuacion, weights=pesos * (1 - y), minlength=n_puntuaciones
        )
        auc = np.sum(positivos * (np.cumsum(negativos) - 0.5 * negativos))
        aucs.append(float(auc / (positivos.sum() * negativos.sum())))
    return {
        "metodo": "bootstrap percentil de grupos de predictores completos, modelo fijo",
        "repeticiones": repeticiones,
        "semilla": semilla,
        "grupos_evaluacion": n_grupos,
        "ic95": np.quantile(aucs, [0.025, 0.975]).tolist(),
        "desviacion_estandar": float(np.std(aucs, ddof=1)),
        "limitacion": "No incorpora variabilidad entre particiones ni reajustes del modelo.",
    }


def estructura_numerica(x: pd.DataFrame) -> dict:
    cuantizacion = {}
    for nombre in x:
        valores = x[nombre].to_numpy()
        paso = 0.01 if nombre == "Amount" else 2.0
        enteros = np.rint(valores / paso).astype(np.int64)
        reconstruidos = enteros * paso
        residuos = np.abs(valores - reconstruidos)
        cuantizacion[nombre] = {
            "paso_examinado": paso,
            "tolerancia_absoluta": 1e-8,
            "fraccion_compatible": float(np.mean(residuos <= 1e-8)),
            "residuo_absoluto_maximo": float(residuos.max()),
            "valores_con_residuo_float_no_cero": int(np.count_nonzero(residuos)),
            "ceros_con_signo_negativo": int(np.sum((valores == 0) & np.signbit(valores))),
            "distintos_originales": int(x[nombre].nunique()),
            "distintos_canonicos": int(np.unique(enteros).size),
            "mcd_enteros_canonicos": int(np.gcd.reduce(np.abs(enteros))),
        }

    conjuntos = {}
    for nombre, columnas in {
        "X1_a_X20": [f"X{i}" for i in range(1, 21)],
        "todos_21_predictores": list(x.columns),
    }.items():
        a = x[columnas].to_numpy()
        media, desviacion = a.mean(axis=0), a.std(axis=0, ddof=1)
        z = (a - media) / desviacion
        correlacion = z.T @ z / (len(z) - 1)
        autovalores = np.linalg.eigvalsh(correlacion)
        i, j = np.triu_indices(len(columnas), k=1)
        indice_maximo = np.argmax(np.abs(correlacion[i, j]))
        conjuntos[nombre] = {
            "rango_numerico_matriz_centrada_estandarizada": int(np.linalg.matrix_rank(z)),
            "numero_columnas": len(columnas),
            "autovalores_correlacion": autovalores.tolist(),
            "numero_condicion_matriz_estandarizada": float(
                np.sqrt(autovalores.max() / autovalores.min())
            ),
            "mayor_correlacion_absoluta": {
                "variables": [columnas[i[indice_maximo]], columnas[j[indice_maximo]]],
                "pearson": float(correlacion[i[indice_maximo], j[indice_maximo]]),
            },
            "media_en_unidades_de_desviacion": dict(zip(columnas, (media / desviacion).tolist())),
            "desviaciones_muestrales": dict(zip(columnas, desviacion.tolist())),
            "desviaciones_decrecientes_en_orden_columnas": bool(np.all(np.diff(desviacion) <= 0)),
            "columnas_identicas": [
                [columnas[k], columnas[l]]
                for k, l in zip(i, j)
                if np.array_equal(a[:, k], a[:, l])
            ],
        }
        if nombre == "todos_21_predictores":
            correlaciones_importe = correlacion[:-1, -1]
            r2 = correlaciones_importe @ np.linalg.solve(
                correlacion[:-1, :-1], correlaciones_importe
            )
            conjuntos[nombre]["relacion_lineal_amount_con_X"] = {
                "r2_descriptivo_en_todo_el_conjunto": float(r2),
                "fraccion_varianza_amount_no_explicada_linealmente": float(1 - r2),
                "interpretacion": "Ajuste descriptivo en los mismos datos, sin estimación de rendimiento predictivo fuera de muestra; Amount no es combinación lineal exacta de X1..X20.",
            }
    return {
        "cuantizacion": cuantizacion,
        "estructura_lineal": conjuntos,
        "limites_interpretacion": [
            "La cuantización describe los valores presentes; no identifica una transformación única ni permite recuperar valores previos.",
            "Centrado y correlación lineal pequeña no implican independencia, normalidad o ausencia de dependencia no lineal.",
            "No se atribuyen significado, unidades, algoritmo de transformación o procedencia a X1..X20.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semilla", type=int, default=20261019)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    inicio = time.monotonic()
    raiz = Path(__file__).resolve().parents[2]
    marcos = [
        pd.read_csv(raiz / "credit_card_construction.csv.zip"),
        pd.read_csv(raiz / "credit_card_production.csv.zip"),
    ]
    x = pd.concat([m.drop(columns="Class") for m in marcos], ignore_index=True)
    y = np.concatenate([np.full(len(m), i, dtype=np.int8) for i, m in enumerate(marcos)])
    assert list(x.columns) == [f"X{i}" for i in range(1, 21)] + ["Amount"]
    assert np.isfinite(x.to_numpy()).all()
    assert np.max(np.abs(x.iloc[:, :20].to_numpy() - np.rint(x.iloc[:, :20].to_numpy() / 2) * 2)) < 1e-8
    assert np.max(np.abs(x["Amount"].to_numpy() - np.rint(x["Amount"].to_numpy() * 100) / 100)) < 1e-8
    grupos_exactos = codificar_grupos(x)
    canonicos = x.copy()
    canonicos.iloc[:, :20] = np.rint(canonicos.iloc[:, :20] / 2)
    canonicos["Amount"] = np.rint(canonicos["Amount"] * 100)
    grupos_canonicos = codificar_grupos(canonicos.astype(np.int64))
    particiones_iguales = bool(np.array_equal(grupos_exactos, grupos_canonicos))
    # Si surgiera una colisión por diferencias de representación, usar el grupo
    # canónico es más conservador: también impide dividir copias casi idénticas.
    grupos = grupos_exactos if particiones_iguales else grupos_canonicos
    separador = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.semilla)
    entrenamiento, evaluacion = next(separador.split(x, y, groups=grupos))
    grupos_comunes = np.intersect1d(grupos[entrenamiento], grupos[evaluacion])
    assert len(grupos_comunes) == 0
    modelo = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.1,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=args.semilla,
    )
    inicio_modelo = time.monotonic()
    modelo.fit(x.iloc[entrenamiento], y[entrenamiento])
    p = modelo.predict_proba(x.iloc[evaluacion])[:, 1]
    duracion_modelo = time.monotonic() - inicio_modelo
    referencia = np.full(len(evaluacion), y[entrenamiento].mean())
    resultado = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "versiones": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "objetivo": "Discriminar archivo de origen: construction=0, production=1; no predecir fraude.",
        "universo": {
            "filas": len(x),
            "construction": len(marcos[0]),
            "production": len(marcos[1]),
            "columnas": list(x.columns),
            "muestreo": "Todos los registros; sin submuestreo.",
            "excluidos": ["Class", "índice", "posición de fila", "nombre de archivo como predictor"],
        },
        "particion": {
            "metodo": "GroupShuffleSplit, 30% de grupos para evaluación",
            "semilla": args.semilla,
            "agrupacion": "igualdad exacta de los 21 predictores" if particiones_iguales else "valores canónicos: X en múltiplos de 2 y Amount en céntimos",
            "grupos_exactos": int(np.unique(grupos_exactos).size),
            "grupos_canonicos": int(np.unique(grupos_canonicos).size),
            "particiones_exacta_y_canonica_identicas": particiones_iguales,
            "grupos_compartidos_train_test": len(grupos_comunes),
            "entrenamiento": {
                "filas": len(entrenamiento),
                "grupos": int(np.unique(grupos[entrenamiento]).size),
                "construction": int(np.sum(y[entrenamiento] == 0)),
                "production": int(np.sum(y[entrenamiento] == 1)),
            },
            "evaluacion": {
                "filas": len(evaluacion),
                "grupos": int(np.unique(grupos[evaluacion]).size),
                "construction": int(np.sum(y[evaluacion] == 0)),
                "production": int(np.sum(y[evaluacion] == 1)),
            },
        },
        "modelo": {"tipo": type(modelo).__name__, "parametros": modelo.get_params()},
        "evaluacion": {
            "roc_auc_origen": float(roc_auc_score(y[evaluacion], p)),
            "bootstrap": bootstrap_auc_grupos(y[evaluacion], p, grupos[evaluacion], args.bootstrap, args.semilla + 1),
            "log_loss": float(log_loss(y[evaluacion], p)),
            "log_loss_referencia_constante": float(log_loss(y[evaluacion], referencia)),
            "brier": float(brier_score_loss(y[evaluacion], p)),
            "brier_referencia_constante": float(brier_score_loss(y[evaluacion], referencia)),
            "segundos_ajuste_y_prediccion": duracion_modelo,
        },
        "limites": [
            "Un AUC próximo a 0.5 no demuestra igualdad de distribuciones; limita la evidencia obtenida con este clasificador y esta partición.",
            "No se ajustaron hiperparámetros usando evaluación; se realizó una única partición predefinida.",
            "El objetivo es origen del archivo. No se entrenó ni evaluó un detector de fraude y Class no interviene.",
            "El orden de filas puede codificar construcción de los archivos: se excluye del modelo y no se interpreta como tiempo.",
            "La partición aleatoria por grupos no es validación temporal; no existe una variable temporal observada.",
        ],
        "estructura_numerica": estructura_numerica(x),
    }
    resultado["segundos_totales"] = time.monotonic() - inicio
    destino = Path(__file__).with_name("deriva_multivariante.json")
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"archivo": str(destino), "evaluacion": resultado["evaluacion"], "particion": resultado["particion"], "segundos_totales": resultado["segundos_totales"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
