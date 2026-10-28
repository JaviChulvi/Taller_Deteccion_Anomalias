"""Vecinos exactos compartidos para LOF y kNN, con scores altos de anomalía.

El preprocesado corresponde al llamador. Las consultas deben contener filas
distintas del conjunto de ajuste: LOF se utiliza aquí en modo ``novelty``.
Solo se conservan matrices de vecinos de tamaño ``n_filas × max_k``.

Los valores de k menores reutilizan prefijos de la búsqueda mayor. Si existen
empates en la distancia del vecino k, una búsqueda independiente de sklearn
puede seleccionar otro subconjunto de esos empates. Las distancias siguen
siendo exactas; el score LOF puede diferir si sus densidades son distintas.
"""

from __future__ import annotations

from operator import index
from time import perf_counter

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from threadpoolctl import threadpool_limits


_EPSILON_LOF = 1e-10


def _entero_positivo(valor, nombre):
    """Valida un entero positivo sin truncar silenciosamente valores reales."""
    try:
        entero = index(valor)
    except TypeError as exc:
        raise ValueError(f"{nombre} debe ser un entero positivo.") from exc
    if isinstance(valor, (bool, np.bool_)) or entero < 1:
        raise ValueError(f"{nombre} debe ser un entero positivo.")
    return entero


def _matriz(X):
    """Conserva todas las filas y exige datos numéricos finitos en float64."""
    return check_array(X, dtype=np.float64, order="C")


def _indice(X, k, threads):
    modelo = NearestNeighbors(
        n_neighbors=k,
        algorithm="brute",
        metric="euclidean",
        n_jobs=threads,
    )
    return modelo.fit(X)


def _densidades_entrenamiento(distancias, indices, k):
    """Densidad de alcanzabilidad local con el mismo epsilon de sklearn."""
    distancia_k = distancias[:, k - 1]
    alcanzabilidad = np.maximum(
        distancias[:, :k], distancia_k[indices[:, :k]]
    )
    densidad = 1.0 / (alcanzabilidad.mean(axis=1) + _EPSILON_LOF)
    return distancia_k, densidad


def _score_lof(distancias, indices, distancia_k, densidad_train):
    alcanzabilidad = np.maximum(distancias, distancia_k[indices])
    densidad_consulta = 1.0 / (
        alcanzabilidad.mean(axis=1) + _EPSILON_LOF
    )
    razones = densidad_train[indices] / densidad_consulta[:, None]
    return razones.mean(axis=1)


class NeighborGeometry:
    """Calcula una vez la geometría exacta reutilizable para varios k.

    ``kneighbors(X=None)`` excluye únicamente la propia fila de entrenamiento.
    Las demás filas con coordenadas iguales permanecen como vecinos legítimos.
    ``metadata_`` recoge tiempos, memoria de las cuatro matrices y empates en
    las fronteras de los k consultados. En max_k no se conoce el siguiente
    vecino, por lo que su recuento de empates se marca como ``None``.
    """

    def __init__(self, X_train, X_query, max_k=60, threads=2):
        self.max_k = _entero_positivo(max_k, "max_k")
        self.threads = _entero_positivo(threads, "threads")
        inicio = perf_counter()
        X_train = _matriz(X_train)
        X_query = _matriz(X_query)
        if X_train.shape[1] != X_query.shape[1]:
            raise ValueError("Train y consulta deben tener las mismas columnas.")
        if self.max_k >= X_train.shape[0]:
            raise ValueError("max_k debe ser menor que el número de filas de train.")

        with threadpool_limits(limits=self.threads):
            self._index = _indice(X_train, self.max_k, self.threads)
            inicio_train = perf_counter()
            self.train_distances_, self.train_indices_ = self._index.kneighbors()
            segundos_train = perf_counter() - inicio_train
            inicio_query = perf_counter()
            self.query_distances_, self.query_indices_ = self._index.kneighbors(
                X_query
            )
            segundos_query = perf_counter() - inicio_query

        self._densidades = {}
        matrices = (
            self.train_distances_,
            self.train_indices_,
            self.query_distances_,
            self.query_indices_,
        )
        self.metadata_ = {
            "algorithm": "brute",
            "metric": "euclidean",
            "dtype": "float64",
            "threads": self.threads,
            "n_train": int(X_train.shape[0]),
            "n_query": int(X_query.shape[0]),
            "n_features": int(X_train.shape[1]),
            "max_k": self.max_k,
            "neighbor_policy": "búsqueda max_k y selección del prefijo de k vecinos",
            "train_neighbors_seconds": segundos_train,
            "query_neighbors_seconds": segundos_query,
            "geometry_seconds": perf_counter() - inicio,
            "neighbor_arrays_bytes": sum(m.nbytes for m in matrices),
            "boundary_ties": {},
        }

    def scores(self, family, k):
        """Devuelve el score por fila de consulta: ``lof`` o distancia ``knn``."""
        if family not in ("lof", "knn"):
            raise ValueError("family debe ser 'lof' o 'knn'.")
        k = _entero_positivo(k, "k")
        if k > self.max_k:
            raise ValueError(f"k={k} supera max_k={self.max_k}.")

        if k < self.max_k:
            empates = {
                "train": int(
                    np.count_nonzero(
                        self.train_distances_[:, k - 1]
                        == self.train_distances_[:, k]
                    )
                ),
                "query": int(
                    np.count_nonzero(
                        self.query_distances_[:, k - 1]
                        == self.query_distances_[:, k]
                    )
                ),
            }
        else:
            empates = {"train": None, "query": None}
        self.metadata_["boundary_ties"][str(k)] = empates

        distancias = self.query_distances_[:, :k]
        if family == "knn":
            return distancias.mean(axis=1)
        if k not in self._densidades:
            self._densidades[k] = _densidades_entrenamiento(
                self.train_distances_, self.train_indices_, k
            )
        distancia_k, densidad = self._densidades[k]
        return _score_lof(
            distancias, self.query_indices_[:, :k], distancia_k, densidad
        )


class NeighborDetector:
    """Estimador serializable para reajustar y puntuar el modelo seleccionado.

    ``params`` contiene ``family`` (``lof`` o ``knn``), ``k`` y, opcionalmente,
    ``max_k`` (60 por defecto). Reutiliza la política del benchmark: busca
    max_k vecinos y toma los primeros k, incluso al reajustar el modelo final.
    En conjuntos pequeños limita max_k a n_train - 1, sin permitir k mayores.
    Así mantiene la misma selección entre empates que ``NeighborGeometry``.

    El score crece con la anomalía. LOF utiliza la fórmula de
    ``-LocalOutlierFactor(novelty=True).score_samples``, con la salvedad de la
    selección entre empates documentada en este módulo. No filtra por etiquetas.
    """

    def __init__(self, params=None, random_state=42, threads=2):
        self.params = {} if params is None else dict(params)
        self.random_state = random_state
        self.threads = threads

    def fit(self, X, y=None):
        inicio = perf_counter()
        self.family_ = self.params.get("family", "lof")
        if self.family_ not in ("lof", "knn"):
            raise ValueError("family debe ser 'lof' o 'knn'.")
        self.k_ = _entero_positivo(self.params.get("k", 20), "k")
        max_k_solicitado = _entero_positivo(
            self.params.get("max_k", 60), "max_k"
        )
        self.threads_ = _entero_positivo(self.threads, "threads")
        X = _matriz(X)
        self.max_k_ = min(max_k_solicitado, X.shape[0] - 1)
        if self.k_ > self.max_k_:
            raise ValueError("k supera max_k o los vecinos disponibles en train.")

        self.n_features_in_ = X.shape[1]
        self.distance_k_ = None
        self.train_density_ = None
        with threadpool_limits(limits=self.threads_):
            self._index = _indice(X, self.max_k_, self.threads_)
            distancias, indices = self._index.kneighbors()
            if self.family_ == "lof":
                self.distance_k_, self.train_density_ = _densidades_entrenamiento(
                    distancias, indices, self.k_
                )

        self.training_info_ = {
            "family": self.family_,
            "k": self.k_,
            "max_k": self.max_k_,
            "requested_max_k": max_k_solicitado,
            "neighbor_policy": "búsqueda max_k y selección del prefijo de k vecinos",
            "algorithm": "brute",
            "metric": "euclidean",
            "dtype": "float64",
            "threads": self.threads_,
            "n_train": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "fit_seconds": perf_counter() - inicio,
            "random_state": self.random_state,
            "uses_labels": False,
            "score_orientation": "mayor significa más anomalía",
        }
        return self

    def score_samples(self, X):
        """Puntúa nuevas filas sin modificar los datos ni las densidades de train."""
        if not hasattr(self, "training_info_"):
            raise NotFittedError("Ejecuta fit antes de score_samples.")
        X = _matriz(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("La consulta debe tener las mismas columnas que train.")
        with threadpool_limits(limits=self.threads_):
            distancias, indices = self._index.kneighbors(X)
            distancias = distancias[:, : self.k_]
            indices = indices[:, : self.k_]
            if self.family_ == "knn":
                return distancias.mean(axis=1)
            return _score_lof(
                distancias, indices, self.distance_k_, self.train_density_
            )
