"""ECOD oficial por cohorte y adaptación inductiva con ECDF de entrenamiento.

Referencia de la implementación consultada: PyOD 3.6.7, ``pyod.models.ecod``
(Zheng Li y Yue Zhao; licencia BSD-2-Clause):
https://pyod.readthedocs.io/en/latest/_modules/pyod/models/ecod.html
https://pyod.readthedocs.io/en/latest/_modules/pyod/utils/stat_models.html

ECODDetector delega el cálculo al paquete oficial. En cada consulta PyOD combina
train con la cohorte consultada y recalcula ECDF y asimetría: no se deben dividir
los datos de evaluación en lotes suponiendo que sus scores permanecen iguales.
Su protocolo es separado y queda excluido de la selección principal.

FrozenECODDetector es una ADAPTACIÓN, no el ECOD original. Congela ECDF y signo de
asimetría en train. Para cada columna usa colas inclusivas y suavizado:
    p_izquierda(x) = (n_train_menores_o_iguales(x) + a) / (n_train + 2*a)
    p_derecha(x)   = (n_train_mayores_o_iguales(x) + a) / (n_train + 2*a)
con a=0.5 por defecto. Retiene la agregación de colas de ECOD, incluido el caso
de asimetría exactamente cero, que suma las dos log-colas. Los extremos fuera
del soporte son finitos y saturan; no se extrapola su distancia a train.

Ambas clases usan todas las filas recibidas sin etiquetas y devuelven scores
altos para anomalías. No son probabilidades de fraude calibradas.
"""

from importlib.metadata import version
from time import perf_counter

import numpy as np
from joblib import parallel_backend
from scipy.stats import skew
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array
from threadpoolctl import threadpool_limits


def _consulta(X, n_features):
    """Valida valores finitos, dimensión y número de predictores."""
    X = check_array(X, dtype=np.float64)
    if X.shape[1] != n_features:
        raise ValueError(
            f"Se esperaban {n_features} predictores, recibidos {X.shape[1]}."
        )
    return X


class ECODDetector:
    """Wrapper del ECOD oficial; cada llamada puntúa una cohorte completa."""

    main_selection_eligible = False
    cohort_dependent = True

    def __init__(self, params=None, random_state=42, threads=2):
        self.params = dict(params or {})
        if set(self.params) - {"contamination"}:
            raise ValueError("ECODDetector solo admite el parámetro contamination.")
        if not isinstance(threads, (int, np.integer)) or threads < 1:
            raise ValueError("threads debe ser un entero positivo.")
        self.random_state = random_state
        self.threads = int(threads)

    def fit(self, X):
        """Ajusta PyOD con todas las filas, sin filtrar ni consultar etiquetas."""
        from pyod.models.ecod import ECOD

        start = perf_counter()
        X = check_array(X, dtype=np.float64, copy=True)
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        # Instancia nueva: un refit no debe conservar X_train del ajuste anterior.
        self.model_ = ECOD(
            contamination=self.params.get("contamination", 0.1),
            n_jobs=min(self.threads, self.n_features_in_),
        )
        with parallel_backend("threading"), threadpool_limits(limits=self.threads):
            self.model_.fit(X)
        self.training_info_ = {
            "modelo": "ECOD oficial PyOD",
            "pyod_version": version("pyod"),
            "protocolo": "cohorte_train_mas_consulta",
            "seleccion_principal_elegible": False,
            "depende_del_lote_consulta": True,
            "usa_etiquetas": False,
            "filas_fit": self.n_samples_fit_,
            "n_features": self.n_features_in_,
            "contamination": float(self.model_.contamination),
            "contamination_solo_umbral": True,
            "n_jobs": self.model_.n_jobs,
            "determinista": True,
            "random_state_no_utilizado": self.random_state,
            "segundos_fit": perf_counter() - start,
        }
        return self

    def score_samples(self, X):
        """Devuelve riesgo alto; no fraccionar la cohorte para ahorrar memoria."""
        if not hasattr(self, "training_info_"):
            raise NotFittedError("ECODDetector requiere fit antes de puntuar.")
        X = _consulta(X, self.n_features_in_)
        with parallel_backend("threading"), threadpool_limits(limits=self.threads):
            scores = self.model_.decision_function(X)
        return np.asarray(scores, dtype=np.float64)


class FrozenECODDetector:
    """ECDF de train congelada: adaptación con scores invariantes al lote."""

    main_selection_eligible = True
    cohort_dependent = False

    def __init__(self, params=None, random_state=42, threads=2):
        self.params = dict(params or {})
        if set(self.params) - {"smoothing"}:
            raise ValueError("FrozenECODDetector solo admite smoothing.")
        self.smoothing = float(self.params.get("smoothing", 0.5))
        if not np.isfinite(self.smoothing) or self.smoothing <= 0:
            raise ValueError("smoothing debe ser positivo y finito.")
        if not isinstance(threads, (int, np.integer)) or threads < 1:
            raise ValueError("threads debe ser un entero positivo.")
        self.random_state = random_state
        self.threads = int(threads)

    def fit(self, X):
        """Aprende distribuciones y asimetría exclusivamente de las filas X."""
        start = perf_counter()
        X = check_array(X, dtype=np.float64)
        self.n_samples_fit_, self.n_features_in_ = X.shape
        with threadpool_limits(limits=self.threads):
            self.sorted_train_ = np.asfortranarray(np.sort(X, axis=0))
            # Las columnas constantes tienen asimetría indefinida; PyOD usa cero.
            variable = self.sorted_train_[0] != self.sorted_train_[-1]
            self.skewness_ = np.zeros(self.n_features_in_, dtype=np.float64)
            if variable.any():
                self.skewness_[variable] = np.nan_to_num(
                    skew(X[:, variable], axis=0, bias=True)
                )
            self.skew_sign_ = np.sign(self.skewness_)
        self.training_info_ = {
            "modelo": "FrozenECOD adaptación ECDF congelada",
            "protocolo": "inductivo_ecdf_train_congelada",
            "seleccion_principal_elegible": True,
            "depende_del_lote_consulta": False,
            "usa_etiquetas": False,
            "filas_fit": self.n_samples_fit_,
            "n_features": self.n_features_in_,
            "smoothing": self.smoothing,
            "colas": "inclusivas; (conteo + a) / (n_train + 2*a)",
            "asimetria": "scipy.stats.skew bias=True; ajustada solo con train",
            "signos_asimetria_train": self.skew_sign_.tolist(),
            "extremos": "score finito saturado fuera del soporte de train",
            "determinista": True,
            "random_state_no_utilizado": self.random_state,
            "segundos_fit": perf_counter() - start,
        }
        return self

    def score_samples(self, X):
        """Puntúa cada fila sin aprender de ella ni del resto de la consulta."""
        if not hasattr(self, "training_info_"):
            raise NotFittedError("FrozenECODDetector requiere fit antes de puntuar.")
        X = _consulta(X, self.n_features_in_)
        scores = np.zeros(X.shape[0], dtype=np.float64)
        denominator = self.n_samples_fit_ + 2 * self.smoothing
        for j in range(self.n_features_in_):
            values = self.sorted_train_[:, j]
            left_count = np.searchsorted(values, X[:, j], side="right")
            right_count = self.n_samples_fit_ - np.searchsorted(
                values, X[:, j], side="left"
            )
            left = -np.log((left_count + self.smoothing) / denominator)
            right = -np.log((right_count + self.smoothing) / denominator)
            # Para skew no nulo, la cola elegida ya está incluida en el máximo.
            # Para skew exactamente cero, ECOD añade ambas log-colas.
            if self.skew_sign_[j] == 0:
                scores += left + right
            else:
                scores += np.maximum(left, right)
        return scores
