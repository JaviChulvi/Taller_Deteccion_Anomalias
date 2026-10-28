#!/usr/bin/env python3
"""Audita igualdad exacta de predictores y solapes de particiones, sin entrenar.

Ejecucion desde cualquier directorio:
    python javi/estudio_datos/auditar_duplicados.py

Las agrupaciones se hacen sobre los valores de las columnas, no sobre un
resumen hash persistido. Igualdad numerica exacta incluye -0.0 == 0.0 y no
demuestra que dos filas correspondan a la misma transaccion real.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def grupos_exactos(df: pd.DataFrame, columnas: list[str]) -> pd.Series:
    """Pandas verifica igualdad de claves, sin fiarse solo de un hash."""
    return df.groupby(columnas, dropna=False, sort=False).ngroup()


def por_clase(etiquetas: pd.Series, mascara: np.ndarray | pd.Series) -> dict:
    seleccion = etiquetas[np.asarray(mascara)]
    resultado = {}
    for valor, total in etiquetas.value_counts(dropna=False).items():
        n = int(seleccion.isna().sum() if pd.isna(valor) else (seleccion == valor).sum())
        clave = "sin_etiqueta" if pd.isna(valor) else str(int(valor))
        resultado[clave] = {"filas": n, "total_clase": int(total), "porcentaje_clase": 100 * n / int(total)}
    return resultado


def resumen_dataset(df: pd.DataFrame, columnas: list[str]) -> tuple[dict, pd.Series]:
    grupos = grupos_exactos(df, columnas)
    tiene_etiquetas = bool(df["Class"].notna().any())
    tamanos = grupos.value_counts()
    duplicado = grupos.map(tamanos).gt(1)
    clases = pd.DataFrame({
        "etiquetas_distintas": df["Class"].groupby(grupos, sort=False).nunique(dropna=True),
        "fraudes": df["Class"].eq(1).groupby(grupos, sort=False).sum(),
        "legitimas": df["Class"].eq(0).groupby(grupos, sort=False).sum(),
        "filas": df["Class"].groupby(grupos, sort=False).size(),
    })
    contradicciones = clases[clases.etiquetas_distintas > 1]
    fraudes = df["Class"].eq(1)
    fraud_grupos = grupos[fraudes].value_counts()
    posiciones = pd.DataFrame({"grupo": grupos, "fila": np.arange(len(df))})
    distancias = posiciones.groupby("grupo", sort=False)["fila"].diff().dropna()
    resumen = {
        "filas": len(df),
        "filas_con_etiqueta": int(df["Class"].notna().sum()),
        "filas_sin_etiqueta": int(df["Class"].isna().sum()),
        "columnas_igualdad": columnas,
        "patrones_unicos": len(tamanos),
        "filas_redundantes_tras_primera": int((tamanos - 1).sum()),
        "grupos_duplicados": int(tamanos.gt(1).sum()),
        "filas_en_grupos_duplicados": int(duplicado.sum()),
        "porcentaje_filas_en_grupos_duplicados": float(100 * duplicado.mean()),
        "maxima_multiplicidad": int(tamanos.max()),
        "histograma_multiplicidades": {str(int(k)): int(v) for k, v in tamanos.value_counts().sort_index().items()},
        "duplicados_por_clase": por_clase(df["Class"], duplicado),
        "grupos_con_etiquetas_contradictorias": len(contradicciones) if tiene_etiquetas else None,
        "filas_con_etiquetas_contradictorias": int(contradicciones.filas.sum()) if tiene_etiquetas else None,
        "fraudes": int(fraudes.sum()) if tiene_etiquetas else None,
        "patrones_fraudulentos_unicos": len(fraud_grupos) if tiene_etiquetas else None,
        "grupos_con_multiples_fraudes": int(fraud_grupos.gt(1).sum()) if tiene_etiquetas else None,
        "fraudes_redundantes_tras_primero": int((fraud_grupos - 1).sum()) if tiene_etiquetas else None,
        "distancias_entre_repeticiones_en_orden_fichero": {
            "n": len(distancias),
            "minimo": float(distancias.min()) if len(distancias) else None,
            "mediana": float(distancias.median()) if len(distancias) else None,
            "maximo": float(distancias.max()) if len(distancias) else None,
            "repeticiones_en_filas_adyacentes": int(distancias.eq(1).sum()),
        },
        "bloques_deciles_orden_fichero": [],
    }
    for i, posiciones_bloque in enumerate(np.array_split(np.arange(len(df)), 10)):
        bloque = df.iloc[posiciones_bloque]
        resumen["bloques_deciles_orden_fichero"].append({
            "bloque": i + 1,
            "primera_fila_base_cero": int(posiciones_bloque[0]),
            "ultima_fila_base_cero": int(posiciones_bloque[-1]),
            "filas": len(bloque),
            "fraudes": int(bloque["Class"].eq(1).sum()) if tiene_etiquetas else None,
            "filas_pertenecientes_a_duplicados_globales": int(duplicado.iloc[posiciones_bloque].sum()),
        })
    return resumen, grupos


def resumen_cruzado(construccion: pd.DataFrame, produccion: pd.DataFrame, columnas: list[str]) -> dict:
    # Una union sobre las columnas originales confirma igualdad de los valores.
    c = construccion.assign(es_fraude=construccion.Class.eq(1), es_legitima=construccion.Class.eq(0)).groupby(columnas, dropna=False, sort=False).agg(
        filas_construccion=("Class", "size"),
        fraudes_construccion=("es_fraude", "sum"),
        legitimas_construccion=("es_legitima", "sum"),
    ).reset_index()
    p = produccion.groupby(columnas, dropna=False, sort=False).size().rename("filas_produccion").reset_index()
    coincidencias = c.merge(p, on=columnas, how="inner", validate="one_to_one")
    grupos_fraude = coincidencias.fraudes_construccion.gt(0)
    grupos_contradictorios = grupos_fraude & coincidencias.legitimas_construccion.gt(0)
    return {
        "columnas_igualdad": columnas,
        "patrones_presentes_en_ambos": len(coincidencias),
        "filas_construccion_con_coincidencia": int(coincidencias.filas_construccion.sum()),
        "filas_produccion_con_coincidencia": int(coincidencias.filas_produccion.sum()),
        "porcentaje_produccion_con_coincidencia": float(100 * coincidencias.filas_produccion.sum() / len(produccion)),
        "pares_de_filas_coincidentes": int((coincidencias.filas_construccion * coincidencias.filas_produccion).sum()),
        "fraudes_construccion_con_coincidencia": int(coincidencias.fraudes_construccion.sum()),
        "patrones_con_fraude_construccion": int(grupos_fraude.sum()),
        "filas_produccion_coincidentes_con_patron_fraudulento_construccion": int(coincidencias.loc[grupos_fraude, "filas_produccion"].sum()),
        "patrones_cruzados_con_etiquetas_contradictorias_construccion": int(grupos_contradictorios.sum()),
    }


def resumen_split(df: pd.DataFrame, grupos: pd.Series, train: np.ndarray, valid: np.ndarray) -> dict:
    grupos_train = set(grupos.iloc[train].tolist())
    grupos_valid = set(grupos.iloc[valid].tolist())
    solape = grupos_train & grupos_valid
    visto = grupos.iloc[valid].isin(grupos_train).to_numpy()
    etiquetas_valid = df["Class"].iloc[valid].reset_index(drop=True)
    return {
        "filas_entrenamiento": len(train),
        "filas_validacion": len(valid),
        "fraccion_validacion": len(valid) / len(df),
        "fraudes_entrenamiento": int(df["Class"].iloc[train].eq(1).sum()),
        "fraudes_validacion": int(etiquetas_valid.eq(1).sum()),
        "tasa_fraude_entrenamiento": float(df["Class"].iloc[train].eq(1).mean()),
        "tasa_fraude_validacion": float(etiquetas_valid.eq(1).mean()),
        "grupos_entrenamiento": len(grupos_train),
        "grupos_validacion": len(grupos_valid),
        "grupos_solapados": len(solape),
        "filas_validacion_con_predictores_en_entrenamiento": int(visto.sum()),
        "porcentaje_validacion_con_predictores_en_entrenamiento": float(100 * visto.mean()),
        "coincidencias_validacion_por_clase": por_clase(etiquetas_valid, visto),
        "grupos_fraudulentos_solapados": len(set(grupos[df["Class"].eq(1)].tolist()) & solape),
    }


def efecto_amount(df: pd.DataFrame, grupos_x: pd.Series) -> dict:
    importes = df.groupby(grupos_x, sort=False)["Amount"].nunique(dropna=False)
    multiples = set(importes[importes.gt(1)].index.tolist())
    filas = grupos_x.isin(multiples)
    return {
        "patrones_X_con_varios_importes": len(multiples),
        "filas_en_esos_patrones": int(filas.sum()),
        "fraudes_etiquetados_en_esos_patrones": int((filas & df["Class"].eq(1)).sum()) if df["Class"].notna().any() else None,
        "filas_sin_etiqueta_en_esos_patrones": int((filas & df["Class"].isna()).sum()),
        "maximo_importes_distintos_en_un_patron_X": int(importes.max()),
    }


def sensibilidad_cuantizacion(construccion: pd.DataFrame, produccion: pd.DataFrame, conjuntos: dict) -> dict:
    """Comprueba si quitar residuos de representacion descubre coincidencias."""
    canonicos = []
    errores = {}
    x = conjuntos["X1_X20"]
    for nombre, df in [("construccion", construccion), ("produccion", produccion)]:
        valores_x = df[x].to_numpy() / 2
        centimos = df["Amount"].to_numpy() * 100
        error_x = float(np.max(np.abs(valores_x - np.rint(valores_x))))
        error_amount = float(np.max(np.abs(centimos - np.rint(centimos))))
        # No se normaliza si el supuesto de reticula no se cumple.
        assert np.isfinite(valores_x).all() and np.isfinite(centimos).all()
        assert error_x < 1e-8 and error_amount < 1e-8
        errores[nombre] = {"max_error_X_dividido_2": error_x, "max_error_Amount_por_100": error_amount}
        canonico = pd.DataFrame(np.rint(valores_x).astype("int64"), columns=x)
        canonico["Amount"] = np.rint(centimos).astype("int64")
        canonicos.append(canonico)
    datasets = [
        ("construccion", construccion, canonicos[0]),
        ("produccion", produccion, canonicos[1]),
        ("combinado", pd.concat([construccion, produccion], ignore_index=True), pd.concat(canonicos, ignore_index=True)),
    ]
    equivalencia = {}
    for nombre, original, canonico in datasets:
        equivalencia[nombre] = {}
        for criterio, columnas in conjuntos.items():
            g_original = grupos_exactos(original, columnas)
            g_canonico = grupos_exactos(canonico, columnas)
            equivalencia[nombre][criterio] = {
                "particion_en_grupos_identica": bool(np.array_equal(g_original.to_numpy(), g_canonico.to_numpy())),
                "patrones_originales": int(g_original.nunique()),
                "patrones_canonicos": int(g_canonico.nunique()),
                "duplicados_redundantes_originales": int(original.duplicated(columnas).sum()),
                "duplicados_redundantes_canonicos": int(canonico.duplicated(columnas).sum()),
            }
    return {
        "metodo": "En copias en memoria: X pasa a round(X/2) y Amount a round(100*Amount), ambos int64, tras comprobar error maximo <1e-8. Se comparan IDs de grupo por orden de primera aparicion para todas las filas, no solo los recuentos.",
        "errores": errores,
        "equivalencia": equivalencia,
        "interpretacion": "Cuando la particion combinada es identica, no cambian tampoco las coincidencias cruzadas ni los solapes de validacion; esta sensibilidad no equivale a buscar similitud aproximada entre transacciones.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    raiz = Path(__file__).resolve().parents[2]
    parser.add_argument("--datos", type=Path, default=raiz)
    parser.add_argument("--salida", type=Path, default=Path(__file__).with_name("duplicados_validacion.json"))
    args = parser.parse_args()
    construccion = pd.read_csv(args.datos / "credit_card_construction.csv.zip")
    produccion = pd.read_csv(args.datos / "credit_card_production.csv.zip")
    x = [f"X{i}" for i in range(1, 21)]
    conjuntos = {"X1_X20": x, "X1_X20_Amount": x + ["Amount"]}
    indices = np.arange(len(construccion))
    train, valid = train_test_split(indices, test_size=0.3, random_state=42, stratify=construccion["Class"])
    resultado = {
        "versiones": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "metodo": "Igualdad numerica exacta por columnas mediante groupby/merge; no se comparan solamente hashes. -0.0 y 0.0 son iguales. NaN en predictores se agrupa conjuntamente.",
        "advertencias": [
            "Dos filas con predictores iguales no prueban identidad de transaccion; la discretizacion puede producir coincidencias.",
            "Class de produccion no permite contrastar etiquetas: esta completamente ausente.",
            "El numero de fila indica orden de archivo, no tiempo ni identidad de cliente.",
            "El solape puede inflar la validacion; esta auditoria no entrena modelos y no cuantifica su efecto en metricas.",
            "La particion agrupada es un analisis de sensibilidad por patron observable, no un sustituto de grupos de cliente o tiempo si existen.",
        ],
        "split_original": {"test_size": 0.3, "random_state": 42, "stratify": "Class"},
        "duplicados_fila_completa_construccion": int(construccion.duplicated().sum()),
        "duplicados_fila_completa_produccion": int(produccion.duplicated().sum()),
        "analisis": {},
    }
    for nombre, columnas in conjuntos.items():
        c, gc = resumen_dataset(construccion, columnas)
        p, gp = resumen_dataset(produccion, columnas)
        # Diez folds estratificados por clase y agrupados por predictores.
        # Se reservan tres folds predefinidos, sin elegirlos por rendimiento.
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        folds = list(sgkf.split(np.zeros((len(construccion), 1)), construccion["Class"], groups=gc))
        valid_agrupada = np.sort(np.concatenate([folds[i][1] for i in [0, 1, 2]]))
        train_agrupada = np.setdiff1d(indices, valid_agrupada, assume_unique=True)
        agrupado = resumen_split(construccion, gc, train_agrupada, valid_agrupada)
        assert agrupado["grupos_solapados"] == 0
        assert len(train_agrupada) + len(valid_agrupada) == len(construccion)
        resultado["analisis"][nombre] = {
            "construccion": c,
            "produccion": p,
            "coincidencias_entre_datasets": resumen_cruzado(construccion, produccion, columnas),
            "split_aleatorio_notebook": resumen_split(construccion, gc, train, valid),
            "alternativa_agrupada": {
                "metodo": "StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42); folds 0,1,2 para validacion y los otros siete para entrenamiento",
                **agrupado,
            },
        }
        if nombre == "X1_X20":
            resultado["efecto_Amount"] = {
                "construccion": efecto_amount(construccion, gc),
                "produccion": efecto_amount(produccion, gp),
                "combinado": efecto_amount(pd.concat([construccion, produccion], ignore_index=True), grupos_exactos(pd.concat([construccion, produccion], ignore_index=True), x)),
            }
    resultado["sensibilidad_cuantizacion"] = sensibilidad_cuantizacion(construccion, produccion, conjuntos)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Auditoria guardada: {args.salida}")
    for nombre, datos in resultado["analisis"].items():
        print(nombre, json.dumps({
            "construccion_duplicados": datos["construccion"]["filas_redundantes_tras_primera"],
            "produccion_duplicados": datos["produccion"]["filas_redundantes_tras_primera"],
            "cruzados": datos["coincidencias_entre_datasets"],
            "split_original": datos["split_aleatorio_notebook"],
            "split_agrupado": datos["alternativa_agrupada"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
