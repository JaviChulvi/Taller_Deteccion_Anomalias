"""Modelos de clase y contrato común: un score alto indica mayor sospecha.

El escalado se ajusta exclusivamente sobre las filas entregadas a fit. Los
clasificadores usan ambas etiquetas; los detectores reciben el régimen elegido
por el evaluador. No se recorta el conjunto de entrenamiento.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CONFIGURACIONES = {
    "constante": [{}],
    "univariante": [{}],
    "gmm_fijo": [{"n_components": 5, "covariance_type": "full", "n_init": 1,
                  "reg_covar": 1e-6, "max_iter": 1000}],
    "gmm": [{"n_components": n, "covariance_type": c, "n_init": 2,
             "reg_covar": 1e-5, "max_iter": 1000}
            for n in [1, 2, 5, 10] for c in ["diag", "full"]],
    "isolation_forest": [{"n_estimators": 400, "max_samples": n, "max_features": 1.0}
                         for n in [256, 4096, 1.0]],
    "pca": [{"n_components": n} for n in [2, 5, 10, 15, 20]],
    "lof": [{"k": n} for n in [10, 30, 60]],
    "knn": [{"k": n} for n in [1, 10, 30]],
    "autoencoder": [{"bottleneck": n, "hidden": 32, "weight_decay": 1e-5,
                     "max_epochs": 200, "patience": 10, "batch_size": 1024}
                    for n in [4, 8]],
    "rbig": [{"n_layers": n, "bins": 128} for n in [20, 50]],
    "ecod_frozen": [{}],
    "ecod_cohorte": [{}],
    "logistica": [{"C": c, "class_weight": w}
                  for c in [0.01, 0.1, 1, 10] for w in [None, "balanced"]],
    "random_forest": [{"n_estimators": 400, "min_samples_leaf": n,
                       "class_weight": w, "max_features": "sqrt"}
                      for n in [1, 5, 20] for w in [None, "balanced_subsample"]],
    "lightgbm": [{"n_estimators": 500, "learning_rate": 0.05, "num_leaves": n,
                 "min_child_samples": 50, "reg_lambda": 1.0, "class_weight": w}
                for n in [7, 15, 31] for w in [None, "balanced"]],
}
SUPERVISADOS = {"logistica", "random_forest", "lightgbm", "univariante"}
SIN_ESCALADO = {"constante", "univariante", "isolation_forest", "random_forest",
                "lightgbm", "ecod_frozen", "ecod_cohorte"}


def regimenes(familia):
    """No confundir algoritmo no supervisado con selección sin etiquetas."""
    if familia in SUPERVISADOS:
        return ["supervisado"]
    if familia == "constante":
        return ["referencia"]
    if familia == "gmm_fijo":
        return ["normales"]
    return ["normales", "todos"]


class Modelo:
    """Adaptador serializable; score_samples siempre crece con el riesgo."""

    def __init__(self, familia, parametros, semilla=42, hilos=2):
        self.familia = familia
        self.parametros = dict(parametros)
        self.semilla = semilla
        self.hilos = hilos

    def fit(self, X, y=None):
        p = self.parametros.copy()
        f = self.familia
        self.scaler_ = None if f in SIN_ESCALADO else StandardScaler().fit(X)
        Z = np.asarray(X, dtype=np.float64) if self.scaler_ is None else self.scaler_.transform(X)
        self.training_info_ = {"filas_fit": len(X), "predictores": X.shape[1],
                               "escalado": self.scaler_ is not None}
        with threadpool_limits(limits=self.hilos):
            if f == "constante":
                self.estimador_ = None
            elif f == "univariante":
                resultados = [(average_precision_score(y, signo * Z[:, j]), j, signo)
                              for j in range(Z.shape[1]) for signo in [-1, 1]]
                _, self.variable_, self.signo_ = max(resultados, key=lambda r: (r[0], -r[1], r[2]))
                self.training_info_.update(variable=int(self.variable_), signo=int(self.signo_))
            elif f.startswith("gmm"):
                self.estimador_ = GaussianMixture(random_state=self.semilla, **p).fit(Z)
                self.training_info_.update(convergencia=bool(self.estimador_.converged_),
                                            iteraciones=int(self.estimador_.n_iter_))
                if not self.estimador_.converged_:
                    raise RuntimeError("GMM no convergió: revisar antes de aceptar este ajuste.")
            elif f == "isolation_forest":
                self.estimador_ = IsolationForest(random_state=self.semilla, n_jobs=self.hilos,
                                                  contamination="auto", **p).fit(Z)
                self.training_info_["muestra_por_arbol"] = int(self.estimador_.max_samples_)
            elif f == "pca":
                self.estimador_ = PCA(svd_solver="full", **p).fit(Z)
            elif f in {"lof", "knn"}:
                from .modelos_vecinos import NeighborDetector
                self.estimador_ = NeighborDetector(
                    params={"family": f, **p}, random_state=self.semilla, threads=self.hilos).fit(Z)
            elif f == "autoencoder":
                from .modelo_ae import AutoencoderDetector
                self.estimador_ = AutoencoderDetector(
                    params=p, random_state=self.semilla, threads=self.hilos).fit(Z)
            elif f == "rbig":
                from .modelo_rbig import RBIGDetector
                self.estimador_ = RBIGDetector(
                    params=p, random_state=self.semilla, threads=self.hilos).fit(Z)
            elif f in {"ecod_frozen", "ecod_cohorte"}:
                from .modelo_ecod import ECODDetector, FrozenECODDetector
                cls = FrozenECODDetector if f == "ecod_frozen" else ECODDetector
                self.estimador_ = cls(params=p, random_state=self.semilla, threads=self.hilos).fit(Z)
            elif f == "logistica":
                self.estimador_ = LogisticRegression(
                    max_iter=3000, solver="lbfgs", random_state=self.semilla, **p).fit(Z, y)
                self.training_info_["iteraciones"] = self.estimador_.n_iter_.tolist()
                if np.max(self.estimador_.n_iter_) >= 3000:
                    raise RuntimeError("La regresión logística agotó max_iter.")
            elif f == "random_forest":
                self.estimador_ = RandomForestClassifier(
                    random_state=self.semilla, n_jobs=self.hilos, **p).fit(Z, y)
            elif f == "lightgbm":
                from lightgbm import LGBMClassifier
                self.estimador_ = LGBMClassifier(
                    random_state=self.semilla, n_jobs=self.hilos, verbosity=-1,
                    deterministic=True, force_col_wise=True, **p).fit(Z, y)
            else:
                raise ValueError(f)
        if hasattr(getattr(self, "estimador_", None), "training_info_"):
            self.training_info_["detalle"] = self.estimador_.training_info_
        return self

    def score_samples(self, X):
        Z = np.asarray(X, dtype=np.float64) if self.scaler_ is None else self.scaler_.transform(X)
        f = self.familia
        with threadpool_limits(limits=self.hilos):
            if f == "constante":
                s = np.zeros(len(Z))
            elif f == "univariante":
                s = self.signo_ * Z[:, self.variable_]
            elif f.startswith("gmm") or f == "isolation_forest":
                s = -self.estimador_.score_samples(Z)
            elif f == "pca":
                reconstruccion = self.estimador_.inverse_transform(self.estimador_.transform(Z))
                s = np.mean((Z - reconstruccion) ** 2, axis=1)
            elif f in {"logistica", "random_forest", "lightgbm"}:
                s = self.estimador_.predict_proba(Z)[:, 1]
            else:
                s = self.estimador_.score_samples(Z)
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (len(X),) or not np.isfinite(s).all():
            raise ValueError(f"Scores inválidos: {f}")
        return s
