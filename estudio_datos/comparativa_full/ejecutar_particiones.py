"""Paraleliza las seis particiones sin cambiar el experimento ni sus ajustes.

Ejemplo:
    python -m estudio_datos.comparativa_full.ejecutar_particiones \
        --familia random_forest --regimen supervisado --trabajos 4 --hilos 2

Cada worker escribe sus ajustes mediante las funciones originales y su propio
estado de progreso. Solo el padre consolida las decisiones. Ante un fallo, los
otros workers terminan su ajuste en curso y cancelan los siguientes; las
excepciones se propagan y los ajustes completos quedan disponibles en caché.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback


ORDEN_PARTICIONES = (0, 1, 2, 3, 4, -1)
_cancelacion = None


def _inicializar_worker(cancelacion):
    global _cancelacion
    _cancelacion = cancelacion


def _ejecutar_particion(familia, regimen, outer, hilos, archivo_estado):
    """Replica una iteración exterior usando cargar, ajustar y elegir originales."""
    from . import experimento

    estado = {
        "familia": familia,
        "regimen": regimen,
        "outer": outer,
        "pid": os.getpid(),
        "inicio": time.time(),
        "ajustes_completos": 0,
        "ajustes_totales": 3 * len(experimento.CONFIGURACIONES[familia])
        + int(outer != -1),
    }
    splits = None

    def actualizar(fase, **detalle):
        estado.update(fase=fase, actualizado=time.time(), **detalle)
        experimento.escribir_json(archivo_estado, estado)

    try:
        actualizar("cargando")
        X, y, grupos, splits, protocolo = experimento.cargar()

        def ajustar(inner, config):
            if _cancelacion.is_set():
                raise RuntimeError("Partición cancelada por el fallo de otra tarea.")
            prefijo = f"outer_{outer}"
            if inner != -1:
                prefijo += f"_inner_{inner}"
            actualizar("ajustando", inner=inner, config=config)
            meta = experimento.ajustar(
                familia, regimen, outer, inner, config,
                splits[f"{prefijo}_train"], splits[f"{prefijo}_eval"],
                X, y, grupos, protocolo, hilos,
            )
            estado["ajustes_completos"] += 1
            actualizar("ajuste_completo", inner=inner, config=config)
            return meta

        resultados = []
        for config, _ in enumerate(experimento.CONFIGURACIONES[familia]):
            resultados.append([ajustar(inner, config) for inner in range(3)])
        ganador, tabla = experimento.elegir(resultados)
        decision = {
            "familia": familia,
            "regimen": regimen,
            "outer": outer,
            "ganador": ganador,
            "busqueda": tabla,
            "protocolo": protocolo["id"],
        }
        if outer != -1:
            decision["evaluacion_exterior"] = ajustar(-1, ganador["config"])
        actualizar("completo", config_ganadora=ganador["config"])
        return decision
    except BaseException:
        actualizar("fallido", error=traceback.format_exc())
        raise
    finally:
        if splits is not None:
            splits.close()


def _entero_positivo(valor):
    numero = int(valor)
    if numero < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo.")
    return numero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--familia", required=True, help="Familia del experimento congelado.")
    parser.add_argument("--regimen", required=True, help="Régimen permitido para la familia.")
    parser.add_argument("--trabajos", type=_entero_positivo, default=4,
                        help="Número de procesos simultáneos; por defecto, 4.")
    parser.add_argument("--hilos", type=_entero_positivo, default=2,
                        help="Hilos por ajuste; por defecto, 2.")
    args = parser.parse_args()

    # Se fijan antes de importar NumPy/sklearn y se heredan al crear los workers.
    os.environ.pop("PYTHONPATH", None)
    for nombre in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[nombre] = str(args.hilos)

    from . import experimento

    if args.familia not in experimento.CONFIGURACIONES:
        parser.error(f"Familia desconocida: {args.familia}")
    if args.regimen not in experimento.regimenes(args.familia):
        parser.error(f"Régimen no permitido para {args.familia}: {args.regimen}")

    nombre = f"{args.familia}__{args.regimen}"
    raiz_estado = experimento.SALIDA / "estado_particiones"
    ruta_estado = raiz_estado / f"{nombre}.json"
    directorio_tareas = raiz_estado / nombre / f"pid_{os.getpid()}"
    archivos_tareas = {
        outer: directorio_tareas / f"outer_{outer}.json"
        for outer in ORDEN_PARTICIONES
    }
    decisiones = {}
    inicio = time.time()
    trabajos = min(args.trabajos, len(ORDEN_PARTICIONES))

    def actualizar_estado(fase, **detalle):
        tareas = []
        for outer in ORDEN_PARTICIONES:
            archivo = archivos_tareas[outer]
            tarea = (
                json.loads(archivo.read_text())
                if archivo.exists()
                else {"outer": outer, "fase": "pendiente", "pid": None}
            )
            tareas.append({"archivo_estado": str(archivo), **tarea})
        experimento.escribir_json(ruta_estado, {
            "pid": os.getpid(),
            "familia": args.familia,
            "regimen": args.regimen,
            "fase": fase,
            "inicio": inicio,
            "actualizado": time.time(),
            "trabajos": trabajos,
            "hilos_por_trabajo": args.hilos,
            "orden_particiones": list(ORDEN_PARTICIONES),
            "tareas_totales": len(ORDEN_PARTICIONES),
            "tareas_completas": len(decisiones),
            "particiones_completas": [
                outer for outer in ORDEN_PARTICIONES if outer in decisiones
            ],
            "tareas": tareas,
            **detalle,
        })

    contexto = multiprocessing.get_context("spawn")
    cancelacion = contexto.Event()
    futuros = {}
    actualizar_estado("iniciando")
    print(json.dumps({
        "pid": os.getpid(), "estado_particiones": str(ruta_estado),
        "familia": args.familia, "regimen": args.regimen,
        "trabajos": trabajos, "hilos": args.hilos,
    }), flush=True)

    try:
        with ProcessPoolExecutor(
            max_workers=trabajos,
            mp_context=contexto,
            initializer=_inicializar_worker,
            initargs=(cancelacion,),
        ) as ejecutor:
            try:
                futuros = {
                    ejecutor.submit(
                        _ejecutar_particion, args.familia, args.regimen, outer,
                        args.hilos, archivos_tareas[outer],
                    ): outer
                    for outer in ORDEN_PARTICIONES
                }
                pendientes = set(futuros)
                while pendientes:
                    terminados, pendientes = wait(
                        pendientes, timeout=5, return_when=FIRST_COMPLETED
                    )
                    for futuro in terminados:
                        outer = futuros[futuro]
                        decision = futuro.result()
                        if decision["outer"] != outer:
                            raise RuntimeError("La tarea devolvió otra partición.")
                        decisiones[outer] = decision
                        print(json.dumps({
                            "outer_completo": outer,
                            "tareas_completas": len(decisiones),
                            "tareas_totales": len(ORDEN_PARTICIONES),
                        }), flush=True)
                    actualizar_estado("ejecutando")
            except BaseException:
                cancelacion.set()
                for futuro in futuros:
                    futuro.cancel()
                actualizar_estado("fallido", error=traceback.format_exc())
                raise

        canonicas = [decisiones[outer] for outer in ORDEN_PARTICIONES]
        ruta_decisiones = experimento.SALIDA / "decisiones" / f"{nombre}.json"
        experimento.escribir_json(ruta_decisiones, canonicas)
        actualizar_estado("verificando_cache")
        reconstruidas = experimento.ejecutar_familia(
            args.familia, args.regimen, args.hilos
        )
        if experimento.digest_json(reconstruidas) != experimento.digest_json(canonicas):
            raise RuntimeError("Las decisiones originales difieren de las paralelas.")
        actualizar_estado("completo", decisiones=str(ruta_decisiones),
                          cache_verificada=True)
        print(json.dumps({
            "estado": "completo", "decisiones": str(ruta_decisiones),
            "tareas_completas": len(decisiones), "cache_verificada": True,
        }), flush=True)
    except BaseException:
        actualizar_estado("fallido", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
