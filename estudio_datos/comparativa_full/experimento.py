"""Datos, particiones, ajustes reales y selección anidada del experimento.

Cada NPZ conserva scores de operaciones que no han participado en su ajuste.
Los JSON acreditan datos, código, índices, parámetros, tamaños y tiempos reales.
La caché se reutiliza únicamente cuando coinciden esas identidades.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .modelos import CONFIGURACIONES, Modelo, regimenes

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "estudio_datos" / "comparativa_full" / "resultados"
PREDICTORES = [f"X{i}" for i in range(1, 21)] + ["Amount"]
SEMILLA = 20261020
CODIGO_AL_IMPORTAR = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def escribir_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(f".{os.getpid()}.tmp")
    temporal.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporal.replace(path)


def versiones():
    return {p: importlib.metadata.version(p) for p in
            ["numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm",
             "pyod", "joblib", "threadpoolctl"]}


def preparar():
    """Materializa cinco folds exteriores y tres interiores sin grupos comunes."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    hashes = {n: digest_bytes((RAIZ / f"credit_card_{n}.csv.zip").read_bytes())
              for n in ["construction", "production"]}
    protocolo = {
        "version": 1, "semilla": SEMILLA, "outer_folds": 5, "inner_folds": 3,
        "datos_sha256": hashes, "predictores": PREDICTORES,
        "configuraciones": CONFIGURACIONES,
        "regimenes": {f: regimenes(f) for f in CONFIGURACIONES},
        "seleccion": "AP media interior máxima; empate recall@1%, menor tiempo y nombre.",
        "ecod_cohorte": "Fuera de selección principal: consulta de cada evaluación como lote completo.",
        "cobertura": "Todas las filas de cada training; normales excluye solo Class=1.",
        "preprocesado": "StandardScaler ajustado con las filas fit de cada régimen; árboles y ECOD sin escalar.",
        "limites": "Validación interna exploratoria de datos previamente examinados; no test externo.",
        "versiones": versiones(), "python": platform.python_version(),
    }
    protocolo["id"] = digest_json(protocolo)
    ruta = SALIDA / "protocolo.json"
    if ruta.exists():
        previo = json.loads(ruta.read_text())
        if previo["id"] != protocolo["id"]:
            raise RuntimeError("El protocolo cambió: revisar y versionar resultados antes de continuar.")
        return previo
    train = pd.read_csv(RAIZ / "credit_card_construction.csv.zip")
    prod = pd.read_csv(RAIZ / "credit_card_production.csv.zip")
    assert list(train.columns) == list(prod.columns) == PREDICTORES + ["Class"]
    X = train[PREDICTORES].to_numpy(dtype=np.float64)
    y = train.Class.to_numpy(dtype=np.int8)
    Xp = prod[PREDICTORES].to_numpy(dtype=np.float64)
    assert X.shape == (170884, 21) and Xp.shape == (113923, 21)
    assert np.isfinite(X).all() and np.isfinite(Xp).all()
    assert y.sum() == 295 and prod.Class.isna().all()
    grupos = train.groupby(PREDICTORES, dropna=False, sort=False).ngroup().to_numpy()
    np.save(SALIDA / "X.npy", X)
    np.save(SALIDA / "y.npy", y)
    np.save(SALIDA / "grupos.npy", grupos)
    np.save(SALIDA / "X_produccion.npy", Xp)
    asignaciones = {}
    exteriores = np.full(len(y), -1, dtype=np.int8)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
    for f, (tr, ev) in enumerate(outer.split(X, y, grupos)):
        assert not np.intersect1d(grupos[tr], grupos[ev]).size
        exteriores[ev] = f
        asignaciones[f"outer_{f}_train"] = tr
        asignaciones[f"outer_{f}_eval"] = ev
    for f in list(range(5)) + [-1]:
        base = np.arange(len(y)) if f == -1 else asignaciones[f"outer_{f}_train"]
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEMILLA + 100 + f)
        for j, (tr, ev) in enumerate(inner.split(X[base], y[base], grupos[base])):
            tr, ev = base[tr], base[ev]
            assert not np.intersect1d(grupos[tr], grupos[ev]).size
            asignaciones[f"outer_{f}_inner_{j}_train"] = tr
            asignaciones[f"outer_{f}_inner_{j}_eval"] = ev
    np.savez_compressed(SALIDA / "particiones.npz", **asignaciones)
    pd.DataFrame({"fila": np.arange(len(y)), "grupo": grupos, "Class": y,
                  "fold_exterior": exteriores}).to_csv(SALIDA / "particiones.csv", index=False)
    escribir_json(ruta, protocolo)
    escribir_json(SALIDA / "datos.json", {
        "filas": len(y), "fraudes": int(y.sum()), "normales": int((y == 0).sum()),
        "patrones": int(np.unique(grupos).size), "produccion": len(Xp),
        "folds": [{"fold": f, "filas": int((exteriores == f).sum()),
                   "fraudes": int(y[exteriores == f].sum())} for f in range(5)],
    })
    return protocolo


def cargar():
    protocolo = json.loads((SALIDA / "protocolo.json").read_text())
    if (platform.python_version() != protocolo["python"]
            or versiones() != protocolo["versiones"]):
        raise RuntimeError(
            "El entorno Python/librerías no coincide con el protocolo. "
            "Restaurar las versiones registradas o iniciar otro experimento "
            "antes de cargar datos, reutilizar caché o entrenar."
        )
    preparados = {n: digest_bytes((SALIDA / n).read_bytes()) for n in
                  ["X.npy", "y.npy", "grupos.npy", "X_produccion.npy", "particiones.npz"]}
    sello = SALIDA / "integridad_datos.json"
    if not sello.exists():
        raise RuntimeError("Falta sellar los datos preparados antes de entrenar.")
    if json.loads(sello.read_text()) != preparados:
        raise RuntimeError("Cambió un archivo de datos/particiones preparado.")
    protocolo["integridad_preparados"] = preparados
    return (np.load(SALIDA / "X.npy", mmap_mode="r"), np.load(SALIDA / "y.npy"),
            np.load(SALIDA / "grupos.npy"), np.load(SALIDA / "particiones.npz"), protocolo)


def sellar_datos():
    """Verifica los arrays frente a los ZIP originales antes del primer ajuste."""
    construccion = pd.read_csv(RAIZ / "credit_card_construction.csv.zip")
    produccion = pd.read_csv(RAIZ / "credit_card_production.csv.zip")
    assert np.array_equal(np.load(SALIDA / "X.npy"), construccion[PREDICTORES].to_numpy())
    assert np.array_equal(np.load(SALIDA / "y.npy"), construccion.Class.to_numpy())
    assert np.array_equal(np.load(SALIDA / "X_produccion.npy"), produccion[PREDICTORES].to_numpy())
    esperado = construccion.groupby(PREDICTORES, dropna=False, sort=False).ngroup().to_numpy()
    assert np.array_equal(np.load(SALIDA / "grupos.npy"), esperado)
    preparados = {n: digest_bytes((SALIDA / n).read_bytes()) for n in
                  ["X.npy", "y.npy", "grupos.npy", "X_produccion.npy", "particiones.npz"]}
    sello = SALIDA / "integridad_datos.json"
    if sello.exists() and json.loads(sello.read_text()) != preparados:
        raise RuntimeError("No sobrescribir el sello de datos modificado.")
    escribir_json(sello, preparados)
    return preparados


def metricas(y, scores, filas):
    orden = np.lexsort((filas, -scores))
    m = {"ap": float(average_precision_score(y, scores)),
         "roc_auc": float(roc_auc_score(y, scores)), "filas": len(y),
         "fraudes": int(y.sum())}
    for q in [0.002, 0.01, 0.02]:
        k = int(np.ceil(len(y) * q))
        tp = int(y[orden[:k]].sum())
        key = str(q)
        corte = scores[orden[k-1]]
        m[f"precision_{key}"] = tp / k
        m[f"recall_{key}"] = tp / int(y.sum())
        m[f"k_{key}"] = k
        m[f"tp_{key}"] = tp
        m[f"empatados_{key}"] = int((scores == corte).sum())
    return m


def codigo_modelo(familia):
    files = [Path(__file__).with_name("modelos.py"), Path(__file__)]
    adicional = {"autoencoder": "modelo_ae.py", "rbig": "modelo_rbig.py",
                 "ecod_frozen": "modelo_ecod.py", "ecod_cohorte": "modelo_ecod.py",
                 "lof": "modelos_vecinos.py", "knn": "modelos_vecinos.py"}
    if familia in adicional:
        files.append(Path(__file__).with_name(adicional[familia]))
    actual = {p.name: digest_bytes(p.read_bytes()) for p in files}
    if any(CODIGO_AL_IMPORTAR.get(n) != h for n, h in actual.items()):
        raise RuntimeError("Cambió código con el proceso activo: reiniciar y revisar caché.")
    return actual


def ruta_ajuste(familia, regimen, outer, inner, config):
    return SALIDA / "ajustes" / familia / regimen / f"outer_{outer}" / f"inner_{inner}" / f"{config:02d}"


def identidad(familia, regimen, outer, inner, config, tr, ev, protocolo):
    return {"protocolo": protocolo["id"], "familia": familia, "regimen": regimen,
            "outer": outer, "inner": inner, "config": config,
            "parametros": CONFIGURACIONES[familia][config],
            "semilla": SEMILLA + 1000 * (outer + 1) + 10 * (inner + 1),
            "train_indices": digest_bytes(np.asarray(tr, dtype=np.int64).tobytes()),
            "eval_indices": digest_bytes(np.asarray(ev, dtype=np.int64).tobytes()),
            "codigo": codigo_modelo(familia), "datos": protocolo["integridad_preparados"]}


def leer_cache(base, ident):
    if base.with_suffix(".json").exists() and base.with_suffix(".npz").exists():
        meta = json.loads(base.with_suffix(".json").read_text())
        if meta.get("identidad") == ident and meta.get("estado") == "completo":
            if digest_bytes(base.with_suffix(".npz").read_bytes()) != meta["scores_sha256"]:
                raise RuntimeError(f"Caché corrupta: {base}")
            return meta
    return None


def guardar_ajuste(base, ident, scores, y, ev, grupos, fit_n, seconds, training_info, avisos):
    base.parent.mkdir(parents=True, exist_ok=True)
    tmp = base.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, fila=ev, y=y[ev], grupo=grupos[ev], score=scores)
    tmp.replace(base.with_suffix(".npz"))
    meta = {"estado": "completo", "identidad": ident,
            "metricas": metricas(y[ev], scores, ev), "filas_fit": int(fit_n),
            "tiempos": seconds, "entrenamiento": training_info, "avisos": avisos,
            "scores_sha256": digest_bytes(base.with_suffix(".npz").read_bytes()),
            "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid()}
    escribir_json(base.with_suffix(".json"), meta)
    print(json.dumps({"familia": ident["familia"], "regimen": ident["regimen"],
                      "outer": ident["outer"], "inner": ident["inner"], "config": ident["config"],
                      "ap": meta["metricas"]["ap"], "filas_fit": fit_n, **seconds}), flush=True)
    return meta


def ajustar(familia, regimen, outer, inner, config, tr, ev, X, y, grupos, protocolo, hilos=2):
    fit_idx = tr[y[tr] == 0] if regimen == "normales" else tr
    assert not np.intersect1d(grupos[fit_idx], grupos[ev]).size
    ident = identidad(familia, regimen, outer, inner, config, fit_idx, ev, protocolo)
    base = ruta_ajuste(familia, regimen, outer, inner, config)
    existente = leer_cache(base, ident)
    if existente:
        return existente
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        modelo = Modelo(familia, ident["parametros"], ident["semilla"], hilos)
        modelo.fit(X[fit_idx], y[fit_idx])
        t1 = time.perf_counter()
        scores = modelo.score_samples(X[ev])
        t2 = time.perf_counter()
    return guardar_ajuste(base, ident, scores, y, ev, grupos, len(fit_idx),
                         {"fit_s": t1-t0, "score_s": t2-t1},
                         modelo.training_info_, [str(w.message) for w in capturados])


def elegir(resultados):
    """La mayor AP interior decide; los demás criterios solo resuelven empates."""
    resumen = []
    for c, filas in enumerate(resultados):
        resumen.append({"config": c, "ap_media": float(np.mean([r["metricas"]["ap"] for r in filas])),
                        "ap_sd": float(np.std([r["metricas"]["ap"] for r in filas], ddof=1)),
                        "recall_1pct": float(np.mean([r["metricas"]["recall_0.01"] for r in filas])),
                        "fit_s_media": float(np.mean([r["tiempos"]["fit_s"] for r in filas]))})
    ganador = sorted(resumen, key=lambda r: (-r["ap_media"], -r["recall_1pct"],
                                            r["fit_s_media"], r["config"]))[0]
    return ganador, resumen


def ejecutar_familia(familia, regimen, hilos=2):
    X, y, grupos, splits, protocolo = cargar()
    decisiones = []
    for outer in list(range(5)) + [-1]:
        resultados = []
        for c, _ in enumerate(CONFIGURACIONES[familia]):
            rs = []
            for inner in range(3):
                tr = splits[f"outer_{outer}_inner_{inner}_train"]
                ev = splits[f"outer_{outer}_inner_{inner}_eval"]
                rs.append(ajustar(familia, regimen, outer, inner, c,
                                 tr, ev, X, y, grupos, protocolo, hilos))
            resultados.append(rs)
        ganador, tabla = elegir(resultados)
        decision = {"familia": familia, "regimen": regimen, "outer": outer,
                    "ganador": ganador, "busqueda": tabla, "protocolo": protocolo["id"]}
        if outer != -1:
            meta = ajustar(familia, regimen, outer, -1, ganador["config"],
                           splits[f"outer_{outer}_train"], splits[f"outer_{outer}_eval"],
                           X, y, grupos, protocolo, hilos)
            decision["evaluacion_exterior"] = meta
        decisiones.append(decision)
        escribir_json(SALIDA / "decisiones" / f"{familia}__{regimen}.json", decisiones)
    return decisiones


def ejecutar_vecinos(regimen, hilos=2):
    """Comparte vecinos exactos entre LOF y kNN, sin muestrear filas."""
    from .modelos_vecinos import NeighborGeometry
    X, y, grupos, splits, protocolo = cargar()
    decisiones = {f: [] for f in ["lof", "knn"]}

    def bloque(outer, inner, tr, ev, seleccion=None):
        fit_idx = tr[y[tr] == 0] if regimen == "normales" else tr
        assert not np.intersect1d(grupos[fit_idx], grupos[ev]).size
        metas, pendientes = {}, []
        for f in ["lof", "knn"]:
            metas[f] = []
            cs = range(len(CONFIGURACIONES[f])) if seleccion is None else [seleccion[f]]
            for c in cs:
                ident = identidad(f, regimen, outer, inner, c, fit_idx, ev, protocolo)
                base = ruta_ajuste(f, regimen, outer, inner, c)
                cache = leer_cache(base, ident)
                if cache:
                    metas[f].append((c, cache))
                else:
                    pendientes.append((f, c, ident, base))
        if pendientes:
            t0 = time.perf_counter()
            scaler = StandardScaler().fit(X[fit_idx])
            with threadpool_limits(limits=hilos):
                geometry = NeighborGeometry(scaler.transform(X[fit_idx]), scaler.transform(X[ev]),
                                            max_k=60, threads=hilos)
                t1 = time.perf_counter()
                for f, c, ident, base in pendientes:
                    ts = time.perf_counter()
                    scores = geometry.scores(f, CONFIGURACIONES[f][c]["k"])
                    meta = guardar_ajuste(
                        base, ident, scores, y, ev, grupos, len(fit_idx),
                        {"fit_s": t1-t0, "score_s": time.perf_counter()-ts},
                        {"filas_fit": len(fit_idx), "geometria_compartida": True,
                         "coste_compartido_no_sumar": True, "detalle": geometry.metadata_}, [])
                    metas[f].append((c, meta))
        return {f: [m for c, m in sorted(ms)] for f, ms in metas.items()}

    for outer in list(range(5)) + [-1]:
        rs = {f: [[] for _ in CONFIGURACIONES[f]] for f in ["lof", "knn"]}
        for inner in range(3):
            metas = bloque(outer, inner, splits[f"outer_{outer}_inner_{inner}_train"],
                           splits[f"outer_{outer}_inner_{inner}_eval"])
            for f in rs:
                for c, meta in enumerate(metas[f]):
                    rs[f][c].append(meta)
        gs = {f: elegir(rs[f]) for f in rs}
        exteriores = {}
        if outer != -1:
            exteriores = bloque(outer, -1, splits[f"outer_{outer}_train"],
                                 splits[f"outer_{outer}_eval"],
                                 {f: gs[f][0]["config"] for f in rs})
        for f in rs:
            decision = {"familia": f, "regimen": regimen, "outer": outer,
                        "ganador": gs[f][0], "busqueda": gs[f][1], "protocolo": protocolo["id"]}
            if outer != -1:
                decision["evaluacion_exterior"] = exteriores[f][0]
            decisiones[f].append(decision)
            escribir_json(SALIDA / "decisiones" / f"{f}__{regimen}.json", decisiones[f])
    return decisiones
