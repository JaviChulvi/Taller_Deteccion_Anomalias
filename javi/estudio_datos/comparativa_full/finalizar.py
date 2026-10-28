"""Reajusta cada candidato con todo construcción y prepara la entrega elegida.

Los parámetros y la elección proceden exclusivamente de validación interior.
Producción se puntúa después: nunca decide el modelo ni sus parámetros.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from .experimento import (RAIZ, SALIDA, cargar, codigo_modelo, digest_bytes, digest_json,
                          escribir_json, versiones, SEMILLA)
from .modelos import CONFIGURACIONES, Modelo, regimenes


def ajustar_final(familia, regimen, hilos=2):
    X, y, grupos, particiones, protocolo = cargar()
    particiones.close()
    assert versiones() == protocolo["versiones"] and platform.python_version() == protocolo["python"]
    seleccion = json.loads((SALIDA / "seleccion_final.json").read_text())
    assert seleccion["protocolo"] == protocolo["id"]
    ruta_decision = SALIDA / "decisiones" / f"{familia}__{regimen}.json"
    assert digest_bytes(ruta_decision.read_bytes()) == seleccion["fuentes"]["decisiones"][ruta_decision.name]
    decisiones = json.loads(ruta_decision.read_text())
    decision = next(d for d in decisiones if d["outer"] == -1)
    assert decision["protocolo"] == protocolo["id"]
    config = decision["ganador"]["config"]
    params = CONFIGURACIONES[familia][config]
    fit_idx = np.flatnonzero(y == 0) if regimen == "normales" else np.arange(len(y))
    Xp = np.load(SALIDA / "X_produccion.npy", mmap_mode="r")
    ruta = SALIDA / "finales" / f"{familia}__{regimen}"
    ruta.parent.mkdir(exist_ok=True)
    ident = {"protocolo": protocolo["id"], "datos": protocolo["integridad_preparados"],
             "familia": familia, "regimen": regimen, "config": config, "parametros": params,
             "semilla": SEMILLA + 99999, "hilos": hilos, "codigo": codigo_modelo(familia),
             "codigo_finalizacion": digest_bytes(Path(__file__).read_bytes()),
             "decision": digest_json(decision)}
    if ruta.with_suffix(".json").exists():
        previo = json.loads(ruta.with_suffix(".json").read_text())
        if previo["identidad"] == ident:
            for ext, key in [(".csv", "scores_sha256"), (".joblib", "modelo_sha256")]:
                if digest_bytes(ruta.with_suffix(ext).read_bytes()) != previo[key]:
                    raise RuntimeError(f"Artefacto final alterado: {ruta}{ext}")
            return previo
    inicio = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo = Modelo(familia, params, ident["semilla"], hilos).fit(X[fit_idx], y[fit_idx])
        ajustado = time.perf_counter()
        scores = modelo.score_samples(Xp)
        puntuado = time.perf_counter()
    assert scores.shape == (113923,) and np.isfinite(scores).all()
    # Verificación con filas reales: el resultado debe sobrevivir serialización.
    # kNN: la distancia euclídea por productos escalares puede perder precisión
    # cerca de cero. Se verificaron vecinos idénticos y ranking sin cambios en
    # la muestra; la resta directa confirmó un error absoluto menor de 1e-7.
    tolerancia = {"rtol": 1e-6, "atol": 1e-7 if familia == "knn" else 1e-8}
    delta_orden = None
    joblib.dump(modelo, ruta.with_suffix(".joblib"), compress=3)
    recargado = joblib.load(ruta.with_suffix(".joblib"))
    if familia == "ecod_cohorte":
        # La misma cohorte completa es parte explícita del contrato del original.
        muestra_scores = recargado.score_samples(Xp)
        delta_serial = float(np.max(np.abs(muestra_scores - scores)))
        np.testing.assert_allclose(muestra_scores, scores, **tolerancia)
        invariant = None
    else:
        indices = np.arange(0, len(Xp), max(1, len(Xp) // 257))
        muestra_scores = recargado.score_samples(Xp[indices])
        delta_serial = float(np.max(np.abs(muestra_scores - scores[indices])))
        np.testing.assert_allclose(muestra_scores, scores[indices], **tolerancia)
        invertidos = recargado.score_samples(Xp[indices[::-1]])[::-1]
        delta_orden = float(np.max(np.abs(invertidos - muestra_scores)))
        np.testing.assert_allclose(invertidos, muestra_scores, **tolerancia)
        invariant = True
    pd.DataFrame({"score": scores}).to_csv(ruta.with_suffix(".csv"), index=False)
    meta = {"identidad": ident, "filas_fit": len(fit_idx), "filas_produccion": len(scores),
            "tiempos": {"fit_s": ajustado-inicio, "score_s": puntuado-ajustado},
            "entrenamiento": modelo.training_info_, "avisos": [str(w.message) for w in avisos],
            "scores_sha256": digest_bytes(ruta.with_suffix(".csv").read_bytes()),
            "modelo_sha256": digest_bytes(ruta.with_suffix(".joblib").read_bytes()),
            "invariante_lote": invariant, "delta_serializacion": delta_serial,
            "delta_orden_lote": delta_orden, "tolerancia_verificacion": tolerancia,
            "alcance_invariancia": "Coincidencia numérica dentro de rtol/atol; no identidad bit a bit.",
            "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    escribir_json(ruta.with_suffix(".json"), meta)
    print(f"REFIT COMPLETO {familia}/{regimen}: {len(fit_idx)} filas, "
          f"{len(scores)} scores; fit={ajustado-inicio:.2f}s", flush=True)
    return meta


def finalizar_todos(hilos=2, trabajos=4):
    """Reajusta combinaciones independientes y conserva su orden canónico."""
    seleccion = json.loads((SALIDA / "seleccion_final.json").read_text())
    pares = [(f, r) for f in CONFIGURACIONES for r in regimenes(f)]
    if trabajos == 1:
        modelos = [ajustar_final(f, r, hilos) for f, r in pares]
    else:
        with ProcessPoolExecutor(
            max_workers=trabajos, mp_context=multiprocessing.get_context("spawn")
        ) as ejecutor:
            futuros = [ejecutor.submit(ajustar_final, f, r, hilos) for f, r in pares]
            modelos = [futuro.result() for futuro in futuros]
    escribir_json(SALIDA / "resumen_refits.json", modelos)
    return seleccion, modelos


def entregar(seleccion=None):
    """Acredita la decisión, los 24 reajustes y la correspondencia de la entrega."""
    if seleccion is None:
        seleccion = json.loads((SALIDA / "seleccion_final.json").read_text())
    _, y, _, particiones, protocolo = cargar()
    particiones.close()
    auditoria_cv = json.loads((SALIDA / "auditoria_completitud.json").read_text())
    assert auditoria_cv["estado"] == "completo" and auditoria_cv["protocolo"] == protocolo["id"]
    assert auditoria_cv["familias_regimenes"] == 24
    assert auditoria_cv["ajustes_verificados"] == 1542
    assert auditoria_cv["evaluaciones_exteriores"] == 120
    familia, regimen = seleccion["familia"], seleccion["regimen"]
    assert familia != "ecod_cohorte", "La cohorte no participa en la selección principal."
    assert seleccion["protocolo"] == protocolo["id"]
    assert seleccion["parametros"] == CONFIGURACIONES[familia][seleccion["config"]]
    verificados = []
    for f in CONFIGURACIONES:
        for r in regimenes(f):
            base = SALIDA / "finales" / f"{f}__{r}"
            meta = json.loads(base.with_suffix(".json").read_text())
            ruta_decision = SALIDA / "decisiones" / f"{f}__{r}.json"
            assert digest_bytes(ruta_decision.read_bytes()) == seleccion["fuentes"]["decisiones"][ruta_decision.name]
            decision = next(d for d in json.loads(ruta_decision.read_text()) if d["outer"] == -1)
            assert meta["identidad"]["protocolo"] == protocolo["id"]
            assert meta["identidad"]["decision"] == digest_json(decision)
            assert meta["identidad"]["familia"] == f and meta["identidad"]["regimen"] == r
            assert meta["identidad"]["config"] == decision["ganador"]["config"]
            assert meta["identidad"]["parametros"] == CONFIGURACIONES[f][decision["ganador"]["config"]]
            assert meta["identidad"]["codigo"] == codigo_modelo(f)
            assert meta["identidad"]["codigo_finalizacion"] == digest_bytes(Path(__file__).read_bytes())
            assert meta["identidad"]["datos"] == protocolo["integridad_preparados"]
            assert meta["filas_fit"] == (int((y == 0).sum()) if r == "normales" else len(y))
            assert meta["filas_produccion"] == 113923
            assert digest_bytes(base.with_suffix(".csv").read_bytes()) == meta["scores_sha256"]
            assert digest_bytes(base.with_suffix(".joblib").read_bytes()) == meta["modelo_sha256"]
            df = pd.read_csv(base.with_suffix(".csv"))
            assert df.shape == (113923, 1) and list(df.columns) == ["score"]
            assert np.isfinite(df.score.to_numpy()).all()
            if f != "ecod_cohorte":
                assert meta["invariante_lote"] is True
            verificados.append({"familia": f, "regimen": r, "filas_fit": meta["filas_fit"],
                                "scores_sha256": meta["scores_sha256"]})
    fuente = SALIDA / "finales" / f"{familia}__{regimen}.csv"
    entrega = RAIZ / "entrega_scores.csv"
    entrega.write_bytes(fuente.read_bytes())
    assert digest_bytes(entrega.read_bytes()) == digest_bytes(fuente.read_bytes())
    resultado = {
        "estado": "completo", "familia": familia, "regimen": regimen,
        "parametros": seleccion["parametros"], "filas": 113923, "columnas": ["score"],
        "mayor_score": "mayor sospecha de fraude; no probabilidad necesariamente calibrada",
        "ruta_entrega": str(entrega.relative_to(RAIZ)),
        "modelo": str((SALIDA / "finales" / f"{familia}__{regimen}.joblib").relative_to(RAIZ)),
        "orden": "Mismo orden de X_produccion.npy, verificado frente al ZIP de producción.",
        "scores_sha256": digest_bytes(entrega.read_bytes()),
        "seleccion_sha256": digest_bytes((SALIDA / "seleccion_final.json").read_bytes()),
        "auditoria_cv_sha256": digest_bytes((SALIDA / "auditoria_completitud.json").read_bytes()),
        "protocolo": protocolo["id"], "refits_verificados": verificados,
        "refit_final_verificado": True, "produccion_verificada": True,
        "rendimiento_produccion": "Desconocido: no hay etiquetas de producción.",
    }
    # La auditoría CV se conserva intacta; este archivo acredita la fase posterior.
    escribir_json(SALIDA / "auditoria_entrega.json", resultado)
    escribir_json(SALIDA / "estado_entrega.json",
                  {"estado": "completo", "auditoria": "auditoria_entrega.json",
                   "scores_sha256": resultado["scores_sha256"]})
    return resultado


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--familia")
    parser.add_argument("--regimen")
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--trabajos", type=int, default=4)
    args = parser.parse_args()
    if args.familia:
        ajustar_final(args.familia, args.regimen, args.hilos)
    else:
        finalizar_todos(args.hilos, args.trabajos)
