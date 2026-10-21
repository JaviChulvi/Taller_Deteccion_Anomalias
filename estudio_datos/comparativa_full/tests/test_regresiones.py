"""Regresiones de reanudación y vecinos, sin usar ni alterar ajustes históricos."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from estudio_datos.comparativa_full import experimento as e
from estudio_datos.comparativa_full import ejecutar_particiones, finalizar
from estudio_datos.comparativa_full.modelos_vecinos import NeighborDetector, NeighborGeometry


class EntornoYCacheTest(unittest.TestCase):
    def setUp(self):
        temporal = tempfile.TemporaryDirectory()
        self.addCleanup(temporal.cleanup)
        self.salida = Path(temporal.name)
        parche = patch.object(e, "SALIDA", self.salida)
        parche.start()
        self.addCleanup(parche.stop)
        self.versiones = e.versiones()
        protocolo = {"id": "prueba", "python": e.platform.python_version(),
                     "versiones": self.versiones}
        (self.salida / "protocolo.json").write_text(json.dumps(protocolo))
        for nombre, datos in {
            "X.npy": np.arange(16, dtype=float).reshape(8, 2),
            "y.npy": np.tile([0, 1], 4),
            "grupos.npy": np.arange(8),
            "X_produccion.npy": np.ones((3, 2)),
        }.items():
            np.save(self.salida / nombre, datos)
        np.savez(self.salida / "particiones.npz", ejemplo=np.arange(8))
        sello = {p.name: e.digest_bytes(p.read_bytes())
                 for p in self.salida.iterdir() if p.suffix in {".npy", ".npz"}}
        (self.salida / "integridad_datos.json").write_text(json.dumps(sello))

    def test_entorno_coincidente_permite_cargar_y_reutilizar_cache(self):
        X, y, grupos, splits, protocolo = e.cargar()
        splits.close()
        tr, ev = np.arange(4), np.arange(4, 8)
        argumentos = ("constante", "referencia", 0, 0, 0,
                      tr, ev, X, y, grupos, protocolo)
        primero = e.ajustar(*argumentos)
        with patch.object(e.Modelo, "fit", side_effect=AssertionError("Reentrenamiento")):
            segundo = e.ajustar(*argumentos)
        self.assertEqual(primero, segundo)

    def test_python_incompatible_se_rechaza_antes_de_leer_arrays(self):
        with patch.object(e.platform, "python_version", return_value="0.0.0"), \
                patch.object(e, "digest_bytes") as huella, \
                patch.object(e.np, "load") as cargar_array:
            with self.assertRaisesRegex(RuntimeError, "entorno Python/librerías"):
                e.cargar()
            huella.assert_not_called()
            cargar_array.assert_not_called()

    def test_librerias_incompatibles_bloquean_todas_las_rutas(self):
        rutas = {
            "familia": lambda: e.ejecutar_familia("constante", "referencia"),
            "vecinos": lambda: e.ejecutar_vecinos("normales"),
            "particion": lambda: ejecutar_particiones._ejecutar_particion(
                "constante", "referencia", 0, 1, self.salida / "estado.json"),
            "refit": lambda: finalizar.ajustar_final("constante", "referencia"),
        }
        incompatibles = {**self.versiones, "scikit-learn": "0.0.0"}
        for nombre, ruta in rutas.items():
            with self.subTest(ruta=nombre), \
                    patch.object(e, "versiones", return_value=incompatibles), \
                    patch.object(e, "leer_cache") as cache, \
                    patch.object(e.Modelo, "fit") as fit, \
                    patch.object(e, "digest_bytes") as huella:
                with self.assertRaisesRegex(RuntimeError, "entorno Python/librerías"):
                    ruta()
                cache.assert_not_called()
                fit.assert_not_called()
                huella.assert_not_called()

    def test_entorno_correcto_sigue_rechazando_datos_alterados(self):
        np.save(self.salida / "y.npy", np.zeros(8))
        with self.assertRaisesRegex(RuntimeError, "Cambió un archivo"):
            e.cargar()


class VecinosTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261020)
        self.X = rng.normal(size=(90, 4))
        self.consulta = rng.normal(size=(11, 4))

    def test_knn_conserva_scores_y_empates_sin_busqueda_de_train(self):
        X = np.concatenate([self.X, np.repeat(self.X[:1], 8, axis=0)])
        consulta = np.concatenate([self.consulta, self.X[:1]])
        referencia = NearestNeighbors(n_neighbors=60, algorithm="brute").fit(X)
        distancias, _ = referencia.kneighbors(consulta)
        original = NearestNeighbors.kneighbors
        llamadas = []

        def solo_consultas(indice, X=None, *args, **kwargs):
            self.assertIsNotNone(X, "kNN no necesita vecinos de todas las filas de train")
            llamadas.append(len(X))
            return original(indice, X, *args, **kwargs)

        with patch.object(NearestNeighbors, "kneighbors", solo_consultas):
            for k in (1, 10, 30):
                with self.subTest(k=k):
                    modelo = NeighborDetector({"family": "knn", "k": k}, threads=1).fit(X)
                    self.assertIsNone(modelo.train_density_)
                    np.testing.assert_allclose(
                        modelo.score_samples(consulta), distancias[:, :k].mean(axis=1),
                        rtol=1e-12, atol=1e-12)
        self.assertEqual(llamadas, [len(consulta)] * 3)

    def test_lof_mantiene_vecinos_de_train_y_scores_de_sklearn(self):
        referencia = LocalOutlierFactor(
            n_neighbors=60, novelty=True, algorithm="brute", n_jobs=1).fit(self.X)
        esperado = -referencia.score_samples(self.consulta)
        original = NearestNeighbors.kneighbors
        llamadas = []

        def registrar(indice, X=None, *args, **kwargs):
            llamadas.append("train" if X is None else "consulta")
            return original(indice, X, *args, **kwargs)

        with patch.object(NearestNeighbors, "kneighbors", registrar):
            modelo = NeighborDetector({"family": "lof", "k": 60}, threads=1).fit(self.X)
            np.testing.assert_allclose(modelo.score_samples(self.consulta), esperado,
                                       rtol=1e-12, atol=1e-12)
        self.assertEqual(llamadas, ["train", "consulta"])


class MetadatosNotebookTest(unittest.TestCase):
    def test_desglose_acepta_ajustes_individuales_compartidos_y_mixtos(self):
        ruta = Path(__file__).resolve().parents[3] / "02_comparativa_modelos.ipynb"
        celdas = json.loads(ruta.read_text())["cells"]
        fuente = next("".join(c["source"]) for c in celdas
                      if c["cell_type"] == "code"
                      and "tabla_desglose_vecinos = " in "".join(c["source"]))
        rng = np.random.default_rng(7)
        X, consulta = rng.normal(size=(90, 21)), rng.normal(size=(11, 21))
        individual = e.Modelo("lof", {"k": 10}, hilos=1).fit(X).training_info_
        geometria = NeighborGeometry(X, consulta, max_k=60, threads=1)
        compartido = {"geometria_compartida": True, "detalle": geometria.metadata_}
        salida = Path("resultados_sinteticos")
        for tipo in ("individual", "compartido", "mixto"):
            with self.subTest(tipo=tipo):
                filas, metadatos = [], {}
                for indice in range(10):
                    regimen, outer = ("normales", indice) if indice < 5 else ("todos", indice-5)
                    fuente_ajuste = f"ajustes/lof/{regimen}/outer_{outer}/00"
                    identidad = {"familia": "lof", "regimen": regimen, "outer": outer,
                                 "inner": -1, "config": 0}
                    compartida = tipo == "compartido" or (tipo == "mixto" and indice % 2 == 0)
                    metadatos[(salida / fuente_ajuste).with_suffix(".json")] = {
                        "identidad": identidad,
                        "entrenamiento": compartido if compartida else individual,
                        "tiempos": {"fit_s": 1.0, "score_s": 2.0},
                    }
                    filas.append({**identidad, "fuente_ajuste": fuente_ajuste,
                                  "filas_fit": len(X), "filas": len(consulta),
                                  "fit_s": 1.0, "score_s": 2.0})
                contexto = {"pd": pd, "np": np, "SALIDA": salida,
                            "por_fold": pd.DataFrame(filas), "leer_json": metadatos.__getitem__,
                            "Markdown": lambda texto: texto, "display": lambda objeto: None}
                exec(compile(fuente, str(ruta), "exec"), contexto)
                tabla = contexto["tabla_desglose_vecinos"]
                self.assertEqual(len(tabla), 10)
                desgloses = ["train_neighbors_seconds", "query_neighbors_seconds", "geometry_seconds"]
                self.assertTrue(tabla.loc[~tabla.geometria_compartida, desgloses].isna().all().all())
                self.assertTrue(tabla.loc[tabla.geometria_compartida, desgloses].notna().all().all())
                np.testing.assert_array_equal(tabla.fit_s, np.ones(10))
                np.testing.assert_array_equal(tabla.score_s, np.full(10, 2.0))


if __name__ == "__main__":
    unittest.main()
