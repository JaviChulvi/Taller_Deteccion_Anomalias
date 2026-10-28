"""Auditoría completa y selección entre familias sin volver a entrenar.

Exige todas las búsquedas interiores y evaluaciones exteriores previstas.
Recalcula métricas desde los scores guardados; nunca calcula una AP conjunta
con scores crudos de modelos de distintos folds. La selección final utiliza
exclusivamente la búsqueda interior sobre construcción (outer=-1).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .experimento import (
    SALIDA, cargar, digest_bytes, elegir, escribir_json, identidad, metricas,
    ruta_ajuste,
)
from .modelos import CONFIGURACIONES, regimenes


FOLDS = (0, 1, 2, 3, 4)
BUSQUEDAS = (*FOLDS, -1)
BASELINE = ("gmm_fijo", "normales")
LIMITE = (
    "Validación interna exploratoria de datos previamente examinados; no es "
    "un test independiente. AP resumida por fold, sin concatenar scores crudos. "
    "La evaluación del procedimiento selecciona familia y configuración solo "
    "con los folds interiores de cada entrenamiento exterior."
)
REGLA = (
    "AP media interior máxima; únicamente ante igualdad exacta: mayor "
    "recall@1%, menor tiempo medio de ajuste, nombre de familia, régimen "
    "e índice de configuración. ECOD por cohorte no es elegible."
)


class AnalisisIncompletoError(RuntimeError):
    """Faltan artefactos; no se puede publicar una comparación parcial."""


def _exigir(condicion, mensaje):
    if not condicion:
        raise RuntimeError(mensaje)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return digest_bytes(Path(path).read_bytes())


def _pares():
    return [(familia, regimen) for familia in CONFIGURACIONES
            for regimen in regimenes(familia)]


def _inventario():
    """Comprueba integridad de la cobertura antes de analizar resultados."""
    faltas, decisiones = [], {}
    for nombre in ("protocolo.json", "integridad_datos.json", "X.npy", "y.npy",
                   "grupos.npy", "X_produccion.npy", "particiones.npz"):
        if not (SALIDA / nombre).is_file():
            faltas.append(nombre)
    for familia, regimen in _pares():
        ruta = SALIDA / "decisiones" / f"{familia}__{regimen}.json"
        if not ruta.is_file():
            faltas.append(str(ruta.relative_to(SALIDA)))
        else:
            registros = _json(ruta)
            _exigir(isinstance(registros, list), f"Decisiones inválidas: {ruta}")
            ids = [r.get("outer") for r in registros]
            _exigir(len(ids) == len(set(ids)), f"Decisiones duplicadas: {ruta}")
            _exigir(set(ids).issubset(BUSQUEDAS), f"Folds inesperados: {ruta}")
            decisiones[familia, regimen] = {r["outer"]: r for r in registros}
            for outer in BUSQUEDAS:
                if outer not in ids:
                    faltas.append(f"{ruta.name}: decisión outer={outer}")
        for outer in BUSQUEDAS:
            for inner in range(3):
                for config in range(len(CONFIGURACIONES[familia])):
                    base = ruta_ajuste(familia, regimen, outer, inner, config)
                    for extension in (".json", ".npz"):
                        path = base.with_suffix(extension)
                        if not path.is_file():
                            faltas.append(str(path.relative_to(SALIDA)))
        for outer, decision in decisiones.get((familia, regimen), {}).items():
            if outer == -1:
                continue
            config = decision.get("ganador", {}).get("config")
            _exigir(type(config) is int and 0 <= config < len(CONFIGURACIONES[familia]),
                    f"Configuración exterior inválida: {ruta}, outer={outer}")
            base = ruta_ajuste(familia, regimen, outer, -1, config)
            for extension in (".json", ".npz"):
                path = base.with_suffix(extension)
                if not path.is_file():
                    faltas.append(str(path.relative_to(SALIDA)))
    if faltas:
        detalle = "\n".join(f"  - {f}" for f in faltas[:15])
        resto = f"\n  ... y {len(faltas)-15} más." if len(faltas) > 15 else ""
        raise AnalisisIncompletoError(
            f"Comparativa incompleta: {len(faltas)} artefactos/decisiones pendientes.\n"
            f"{detalle}{resto}\nNo se generaron tablas ni selección final."
        )
    return decisiones


def _comparar(actual, esperado, contexto):
    """Contrasta JSON numérico, tolerando solo redondeo de coma flotante."""
    if isinstance(esperado, dict):
        _exigir(isinstance(actual, dict) and set(actual) == set(esperado),
                f"Claves inconsistentes en {contexto}")
        for key, value in esperado.items():
            _comparar(actual[key], value, f"{contexto}.{key}")
    elif isinstance(esperado, list):
        _exigir(isinstance(actual, list) and len(actual) == len(esperado),
                f"Longitud inconsistente en {contexto}")
        for i, (a, b) in enumerate(zip(actual, esperado)):
            _comparar(a, b, f"{contexto}[{i}]")
    elif isinstance(esperado, (float, np.floating)):
        _exigir(isinstance(actual, (int, float)) and np.isfinite(actual)
                and np.isclose(actual, esperado, atol=1e-12, rtol=1e-12),
                f"Valor inconsistente en {contexto}: {actual!r} != {esperado!r}")
    else:
        _exigir(actual == esperado, f"Valor inconsistente en {contexto}")


def _particiones(splits, y, grupos):
    """Acredita cobertura completa y ausencia de grupos compartidos."""
    todos = np.arange(len(y))
    def comprobar(tr, ev, base, nombre):
        for indices in (tr, ev):
            _exigir(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer),
                    f"Índices inválidos: {nombre}")
            _exigir(len(np.unique(indices)) == len(indices)
                    and np.all((indices >= 0) & (indices < len(y))),
                    f"Índices repetidos/fuera de rango: {nombre}")
        _exigir(np.array_equal(np.sort(np.r_[tr, ev]), np.sort(base)),
                f"Partición no exhaustiva o solapada: {nombre}")
        _exigir(not np.intersect1d(grupos[tr], grupos[ev]).size,
                f"Grupos compartidos: {nombre}")
        _exigir(set(np.unique(y[tr])) == {0, 1}
                and set(np.unique(y[ev])) == {0, 1},
                f"Falta alguna clase: {nombre}")
    externos = []
    for outer in FOLDS:
        tr, ev = splits[f"outer_{outer}_train"], splits[f"outer_{outer}_eval"]
        comprobar(tr, ev, todos, f"outer={outer}")
        externos.append(ev)
    _exigir(np.array_equal(np.sort(np.concatenate(externos)), todos),
            "Las evaluaciones exteriores no cubren cada fila exactamente una vez.")
    for outer in BUSQUEDAS:
        base = todos if outer == -1 else splits[f"outer_{outer}_train"]
        interiores = []
        for inner in range(3):
            tr = splits[f"outer_{outer}_inner_{inner}_train"]
            ev = splits[f"outer_{outer}_inner_{inner}_eval"]
            comprobar(tr, ev, base, f"outer={outer}, inner={inner}")
            interiores.append(ev)
        _exigir(np.array_equal(np.sort(np.concatenate(interiores)), np.sort(base)),
                f"Cobertura interior incompleta: outer={outer}")


def _ajuste(familia, regimen, outer, inner, config, splits, y, grupos, protocolo):
    prefijo = f"outer_{outer}" if inner == -1 else f"outer_{outer}_inner_{inner}"
    tr, ev = splits[f"{prefijo}_train"], splits[f"{prefijo}_eval"]
    fit = tr[y[tr] == 0] if regimen == "normales" else tr
    base = ruta_ajuste(familia, regimen, outer, inner, config)
    meta = _json(base.with_suffix(".json"))
    _exigir(meta.get("estado") == "completo", f"Ajuste incompleto: {base}")
    esperado = identidad(familia, regimen, outer, inner, config, fit, ev, protocolo)
    _exigir(meta.get("identidad") == esperado, f"Identidad de ajuste incompatible: {base}")
    _exigir(meta.get("filas_fit") == len(fit), f"Entrenamiento recortado: {base}")
    _exigir(_sha(base.with_suffix(".npz")) == meta.get("scores_sha256"),
            f"Scores corruptos: {base}")
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as archivo:
        _exigir(set(archivo.files) == {"fila", "y", "grupo", "score"},
                f"Esquema NPZ inesperado: {base}")
        contenido = {k: archivo[k] for k in archivo.files}
    for nombre, valores in (("fila", ev), ("y", y[ev]), ("grupo", grupos[ev])):
        _exigir(np.array_equal(contenido[nombre], valores),
                f"Filas/etiquetas/grupos incompatibles ({nombre}): {base}")
    scores = contenido["score"]
    _exigir(scores.shape == (len(ev),) and np.isfinite(scores).all(),
            f"Scores no finitos o longitud incorrecta: {base}")
    calculadas = metricas(y[ev], scores, ev)
    _comparar(meta["metricas"], calculadas, f"{base}.metricas")
    for nombre in ("fit_s", "score_s"):
        valor = meta["tiempos"][nombre]
        _exigir(np.isfinite(valor) and valor >= 0, f"Tiempo inválido: {base}")
    meta = {**meta, "metricas": calculadas}
    return meta, contenido, str(base.relative_to(SALIDA))


def _clave_seleccion(fila):
    return (-fila["ap_media"], -fila["recall_1pct"], fila["fit_s_media"],
            fila["familia"], fila["regimen"], fila["config"])


def _validar_oof(tabla, y, grupos, nombre):
    tabla = tabla.sort_values("fila", kind="stable").reset_index(drop=True)
    _exigir(len(tabla) == 170884 and np.array_equal(tabla.fila, np.arange(len(y))),
            f"OOF no cubre las 170884 filas exactamente una vez: {nombre}")
    _exigir(np.array_equal(tabla.y, y) and np.array_equal(tabla.grupo, grupos)
            and np.isfinite(tabla.score).all(), f"OOF incompatible: {nombre}")
    return tabla


def _csv(tabla, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(f".{os.getpid()}.tmp")
    tabla.to_csv(temporal, index=False)
    temporal.replace(path)


def analizar():
    """Audita todos los ajustes y escribe tablas solo si la cobertura es completa."""
    decisiones = _inventario()
    X, y, grupos, archivo_splits, protocolo = cargar()
    del X
    with archivo_splits:
        splits = {k: archivo_splits[k] for k in archivo_splits.files}
    _exigir(protocolo["outer_folds"] == 5 and protocolo["inner_folds"] == 3,
            "El protocolo no corresponde a validación anidada 5/3.")
    _exigir(protocolo["configuraciones"] == CONFIGURACIONES
            and protocolo["regimenes"] == {f: regimenes(f) for f in CONFIGURACIONES},
            "Configuraciones o regímenes distintos del protocolo.")
    _exigir(len(y) == len(grupos) == 170884 and set(np.unique(y)) == {0, 1}
            and int(y.sum()) == 295, "Datos de construcción inesperados.")
    _particiones(splits, y, grupos)
    filas_fold, candidatos, oof, fuentes_ajustes = [], [], {}, {}
    exterior_por_clave = {}
    for familia, regimen in _pares():
        fragmentos = []
        for outer in BUSQUEDAS:
            decision = decisiones[familia, regimen][outer]
            _exigir(decision.get("familia") == familia
                    and decision.get("regimen") == regimen
                    and decision.get("protocolo") == protocolo["id"],
                    f"Decisión incompatible: {familia}/{regimen}/{outer}")
            resultados = []
            for config in range(len(CONFIGURACIONES[familia])):
                evaluaciones = []
                for inner in range(3):
                    meta, _, fuente = _ajuste(
                        familia, regimen, outer, inner, config, splits, y, grupos, protocolo)
                    evaluaciones.append(meta)
                    fuentes_ajustes[fuente] = meta["scores_sha256"]
                resultados.append(evaluaciones)
            ganador, busqueda = elegir(resultados)
            _comparar(decision["ganador"], ganador, "decisión.ganador")
            _comparar(decision["busqueda"], busqueda, "decisión.busqueda")
            candidato = {"familia": familia, "regimen": regimen, "outer": outer,
                         "elegible": familia != "ecod_cohorte", **ganador}
            candidatos.append(candidato)
            if outer == -1:
                continue
            meta, contenido, fuente = _ajuste(
                familia, regimen, outer, -1, ganador["config"], splits, y, grupos, protocolo)
            _comparar(decision["evaluacion_exterior"], meta, "decisión.evaluacion_exterior")
            fuentes_ajustes[fuente] = meta["scores_sha256"]
            fila = {"familia": familia, "regimen": regimen, "outer": outer,
                    "config": ganador["config"], "elegible": candidato["elegible"],
                    "ap_interior": ganador["ap_media"], **meta["metricas"],
                    **meta["tiempos"], "filas_fit": meta["filas_fit"],
                    "fuente_ajuste": fuente}
            filas_fold.append(fila)
            exterior_por_clave[familia, regimen, outer] = fila
            fragmentos.append(pd.DataFrame({
                **contenido, "outer": outer, "config": ganador["config"]}))
        oof[familia, regimen] = _validar_oof(
            pd.concat(fragmentos, ignore_index=True), y, grupos, f"{familia}/{regimen}")

    tabla_fold = pd.DataFrame(filas_fold)
    for outer in FOLDS:
        ap_base = exterior_por_clave[*BASELINE, outer]["ap"]
        mascara = tabla_fold.outer.eq(outer)
        tabla_fold.loc[mascara, "ap_baseline_gmm"] = ap_base
        tabla_fold.loc[mascara, "delta_ap_gmm"] = tabla_fold.loc[mascara, "ap"] - ap_base
    resumen = []
    for (familia, regimen), tabla in tabla_fold.groupby(["familia", "regimen"], sort=False):
        _exigir(set(tabla.outer) == set(FOLDS) and len(tabla) == 5,
                f"Faltan evaluaciones exteriores: {familia}/{regimen}")
        fila = {"familia": familia, "regimen": regimen, "folds": 5,
                "filas_oof": len(oof[familia, regimen]), "fraudes_oof": int(y.sum()),
                "elegible": familia != "ecod_cohorte",
                "ap_media": tabla.ap.mean(), "ap_sd": tabla.ap.std(ddof=1),
                "ap_min": tabla.ap.min(), "ap_max": tabla.ap.max(),
                "delta_ap_gmm_media": tabla.delta_ap_gmm.mean(),
                "folds_mejor_gmm": int(tabla.delta_ap_gmm.gt(0).sum()),
                "fit_s_media": tabla.fit_s.mean(), "score_s_media": tabla.score_s.mean()}
        for q in ("0.002", "0.01", "0.02"):
            for metrica in ("precision", "recall"):
                columna = f"{metrica}_{q}"
                fila[f"{columna}_media"] = tabla[columna].mean()
        resumen.append(fila)
    tabla_resumen = pd.DataFrame(resumen).sort_values(
        ["ap_media", "familia", "regimen"], ascending=[False, True, True], kind="stable")
    procedimientos, procedimiento_oof, seleccionados = [], [], {}
    for outer in BUSQUEDAS:
        elegibles = [c for c in candidatos if c["outer"] == outer and c["elegible"]]
        elegido = min(elegibles, key=_clave_seleccion)
        seleccionados[outer] = elegido
        if outer == -1:
            continue
        familia, regimen = elegido["familia"], elegido["regimen"]
        evaluacion = exterior_por_clave[familia, regimen, outer]
        ap_base = exterior_por_clave[*BASELINE, outer]["ap"]
        procedimientos.append({
            **evaluacion, "ap_interior_seleccion": elegido["ap_media"],
            "ap_baseline_gmm": ap_base, "delta_ap_gmm": evaluacion["ap"] - ap_base})
        datos_oof = oof[familia, regimen]
        procedimiento_oof.append(datos_oof.loc[datos_oof.outer.eq(outer)].assign(
            familia=familia, regimen=regimen))
    tabla_procedimiento = pd.DataFrame(procedimientos)
    oof_procedimiento = _validar_oof(
        pd.concat(procedimiento_oof, ignore_index=True), y, grupos, "selector anidado")

    # Solo se escriben entregables después de validar el conjunto completo.
    tablas = {"tabla_por_fold.csv": tabla_fold,
              "tabla_resumen_familia_regimen.csv": tabla_resumen,
              "tabla_candidatos_interiores.csv": pd.DataFrame(candidatos),
              "tabla_procedimiento_anidado.csv": tabla_procedimiento}
    for nombre, tabla in tablas.items():
        _csv(tabla, SALIDA / nombre)
    for (familia, regimen), tabla in oof.items():
        _csv(tabla, SALIDA / "scores_oof" / f"{familia}__{regimen}.csv")
    _csv(oof_procedimiento, SALIDA / "scores_oof" / "procedimiento_anidado.csv")
    final = seleccionados[-1]
    fuentes = {
        "protocolo.json": _sha(SALIDA / "protocolo.json"),
        "integridad_datos.json": _sha(SALIDA / "integridad_datos.json"),
        "analizar.py": _sha(__file__),
        "tabla_candidatos_interiores.csv": _sha(SALIDA / "tabla_candidatos_interiores.csv"),
        "decisiones": {f"{f}__{r}.json": _sha(SALIDA / "decisiones" / f"{f}__{r}.json")
                       for f, r in _pares()},
    }
    seleccion_final = {
        "estado": "seleccionado_pendiente_refit_y_produccion",
        "protocolo": protocolo["id"], "familia": final["familia"],
        "regimen": final["regimen"], "config": final["config"],
        "parametros": CONFIGURACIONES[final["familia"]][final["config"]],
        "regla": REGLA, "criterios_interiores": final,
        "origen_seleccion": "outer=-1; tres folds interiores sobre construcción completa",
        "metricas_exteriores_usadas_para_elegir": False,
        "familias_excluidas_seleccion": ["ecod_cohorte"],
        "filas_refit_esperadas": int((y == 0).sum()) if final["regimen"] == "normales" else len(y),
        "filas_produccion_esperadas": 113923, "fuentes": fuentes, "limites": LIMITE,
    }
    auditoria = {
        "estado": "completo", "protocolo": protocolo["id"],
        "alcance": "Búsquedas interiores y evaluaciones exteriores; refit final y producción se verifican por separado.",
        "refit_final_verificado": False, "produccion_verificada": False,
        "familias_regimenes": len(_pares()), "ajustes_verificados": len(fuentes_ajustes),
        "evaluaciones_exteriores": len(tabla_fold),
        "filas_oof_por_familia_regimen": len(y),
        "ap_procedimiento_media": float(tabla_procedimiento.ap.mean()),
        "ap_procedimiento_sd": float(tabla_procedimiento.ap.std(ddof=1)),
        "delta_ap_gmm_procedimiento_media": float(tabla_procedimiento.delta_ap_gmm.mean()),
        "ap_pooled_calculada": False, "limites": LIMITE,
        "costes_vecinos": "La geometría LOF/kNN es compartida; no sumar sus tiempos como trabajos independientes.",
        "scores_verificados_sha256": fuentes_ajustes, "fuentes": fuentes,
    }
    escribir_json(SALIDA / "auditoria_completitud.json", auditoria)
    escribir_json(SALIDA / "seleccion_final.json", seleccion_final)
    return seleccion_final


def main():
    try:
        seleccion = analizar()
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ANÁLISIS DETENIDO: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps({
        "familia": seleccion["familia"], "regimen": seleccion["regimen"],
        "config": seleccion["config"], "estado": seleccion["estado"],
        "seleccion_final": str(SALIDA / "seleccion_final.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
