"""Ejecutor local reanudable. Ejemplo: python -m ...ejecutar --todos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .experimento import (RAIZ, SALIDA, escribir_json, preparar, ruta_ajuste,
                          ejecutar_familia, ejecutar_vecinos, sellar_datos)
from .modelos import CONFIGURACIONES, regimenes


def fijar_metadatos_vecinos(regimen):
    """Usa la instantánea persistida antes de que cambie la geometría compartida.

    La geometría acumula diagnósticos de empates para cada k. Esa información
    mutable no debe alterar retrospectivamente la evaluación ya guardada de
    otra configuración. Los scores y sus métricas no se modifican.
    """
    for familia in ("lof", "knn"):
        ruta = SALIDA / "decisiones" / f"{familia}__{regimen}.json"
        decisiones = json.loads(ruta.read_text())
        for decision in decisiones:
            if decision["outer"] != -1:
                base = ruta_ajuste(familia, regimen, decision["outer"], -1,
                                   decision["ganador"]["config"])
                decision["evaluacion_exterior"] = json.loads(
                    base.with_suffix(".json").read_text())
        escribir_json(ruta, decisiones)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparar", action="store_true")
    parser.add_argument("--todos", action="store_true")
    parser.add_argument("--familia")
    parser.add_argument("--regimen")
    parser.add_argument("--hilos", type=int, default=2)
    parser.add_argument("--trabajos", type=int, default=4)
    args = parser.parse_args()
    if args.preparar:
        print(json.dumps(preparar(), indent=2, ensure_ascii=False))
        sellar_datos()
        return
    if args.familia:
        try:
            if args.familia == "vecinos":
                ejecutar_vecinos(args.regimen, args.hilos)
                fijar_metadatos_vecinos(args.regimen)
            else:
                ejecutar_familia(args.familia, args.regimen, args.hilos)
        except Exception:
            error = traceback.format_exc()
            escribir_json(SALIDA / "errores" / f"{args.familia}__{args.regimen}.json",
                          {"error": error, "fecha": time.time(), "pid": os.getpid()})
            raise
        return
    if not args.todos:
        parser.error("Usar --preparar, --todos o --familia/--regimen.")
    preparar()
    sellar_datos()
    # Primero referencias y modelos clásicos; todos los trabajos alcanzan sus 5 folds.
    cola = [(f, r) for f in CONFIGURACIONES if f not in {"lof", "knn"}
            for r in regimenes(f)]
    cola += [("vecinos", r) for r in ["normales", "todos"]]
    activos, completos, fallidos = [], [], []
    logs = SALIDA / "logs"
    logs.mkdir(exist_ok=True)
    while cola or activos:
        while cola and len(activos) < args.trabajos:
            f, r = cola.pop(0)
            log = (logs / f"{f}__{r}.log").open("a", buffering=1)
            command = [sys.executable, "-m", "estudio_datos.comparativa_full.ejecutar",
                       "--familia", f, "--regimen", r, "--hilos", str(args.hilos)]
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                env[key] = str(args.hilos)
            p = subprocess.Popen(command, cwd=RAIZ, stdout=log, stderr=subprocess.STDOUT, env=env)
            activos.append((p, f, r, log))
            print(f"INICIO {f}/{r}: pid={p.pid}", flush=True)
        for entry in activos[:]:
            p, f, r, log = entry
            code = p.poll()
            if code is not None:
                activos.remove(entry)
                log.close()
                (completos if code == 0 else fallidos).append([f, r, code])
                print(f"FIN {f}/{r}: exit={code}", flush=True)
        escribir_json(SALIDA / "estado.json", {
            "pid": os.getpid(), "actualizado": time.time(),
            "activos": [{"pid": p.pid, "familia": f, "regimen": r} for p, f, r, _ in activos],
            "pendientes": cola, "completos": completos, "fallidos": fallidos,
        })
        if activos:
            time.sleep(3)
    if fallidos:
        raise SystemExit(f"Hay trabajos por resolver: {fallidos}")


if __name__ == "__main__":
    main()
