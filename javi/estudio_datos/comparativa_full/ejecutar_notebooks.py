"""Ejecuta los notebooks canónicos y conserva sus salidas en el propio fichero."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


RAIZ = Path(__file__).resolve().parents[2]


def ejecutar(nombre):
    ruta = RAIZ / nombre
    notebook = nbformat.read(ruta, as_version=4)

    def guardar():
        temporal = ruta.with_suffix(".ejecutando.ipynb")
        nbformat.write(notebook, temporal)
        temporal.replace(ruta)

    def iniciar(cell, cell_index):
        if cell.cell_type == "code":
            print(f"{nombre}: celda {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def terminado(cell, cell_index, execute_reply):
        guardar()
        print(f"{nombre}: celda {cell_index + 1}, "
              f"estado={execute_reply['content']['status']}", flush=True)

    cliente = NotebookClient(
        notebook, timeout=None, kernel_name="taller-comparativa",
        resources={"metadata": {"path": str(RAIZ)}},
        allow_errors=False, record_timing=True,
        on_cell_start=iniciar, on_cell_executed=terminado,
    )
    notebook.metadata["ejecucion_local"] = {
        "inicio_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "ejecutando",
    }
    try:
        cliente.execute()
        errores = [o for c in notebook.cells for o in c.get("outputs", [])
                   if o.output_type == "error"]
        if errores:
            raise RuntimeError(f"Se encontraron {len(errores)} salidas de error.")
        notebook.metadata.ejecucion_local["estado"] = "completo"
    except Exception:
        notebook.metadata.ejecucion_local["estado"] = "error"
        raise
    finally:
        notebook.metadata.ejecucion_local["fin_utc"] = datetime.now(timezone.utc).isoformat()
        guardar()
    print(f"COMPLETO {ruta}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", default=[
        "02_comparativa_modelos.ipynb", "03_modelo_final_y_entrega.ipynb"
    ])
    args = parser.parse_args()
    for nombre in args.notebooks:
        ejecutar(nombre)
