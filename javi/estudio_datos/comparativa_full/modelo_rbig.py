"""RBIG con histogramas normalizados, rotaciones PCA y logdensidad estable.

Adaptación del algoritmo usado en ``AD_rbig_demo_clase.ipynb`` de MIAX:
https://github.com/IPL-UV/rbig/tree/a9124fc07d8e851833802d604fe552b6327ab980

Se conserva la secuencia histograma marginal -> CDF normal inversa -> PCA.
Esta implementación NO es el paquete original: normaliza las pseudocuentas
antes de integrar la CDF, añade colas exponenciales con masa explícita y calcula
directamente la logdensidad. Las colas evitan recortar la CDF a un intervalo
finito, lo que haría nula la derivada fuera del soporte. No usa los atributos
privados de ``scipy.stats.rv_histogram`` ni estima distribuciones al predecir.

Para la transformación G de entrada a gaussiana se aplica
log p_X(x) = log phi(G(x)) + log |det J_G(x)|. Una rotación PCA sin
blanqueamiento tiene determinante absoluto uno. Cada marginal aporta
log p_marginal(x) - log phi(Phi^-1(F_marginal(x))).

Las diferencias respecto del material son decisiones estadísticas explícitas,
no una promesa de equivalencia numérica con sus resultados guardados. Toda
observación de entrenamiento participa en cada histograma y rotación. El
escalado externo debe ajustarse exclusivamente con el conjunto de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.special import ndtri_exp
from threadpoolctl import threadpool_limits


_LOG_2PI = float(np.log(2.0 * np.pi))
_SOURCE_URL = (
    "https://github.com/IPL-UV/rbig/tree/"
    "a9124fc07d8e851833802d604fe552b6327ab980"
)


@dataclass
class _MarginalHistogram:
    """Histograma continuo por tramos con dos colas exponenciales propias."""

    edges: np.ndarray
    masses: np.ndarray
    cumulative_left: np.ndarray
    cumulative_right: np.ndarray
    log_density: np.ndarray
    tail_mass: float
    left_scale: float
    right_scale: float

    @classmethod
    def fit(cls, values, bins, pseudocount, bound_ext):
        lower = float(np.min(values))
        upper = float(np.max(values))
        span = upper - lower
        if span == 0.0:
            # Una marginal constante necesita suavizado para definir densidad.
            span = max(1.0, abs(lower)) * 1e-3
            lower -= span / 2.0
            upper += span / 2.0
        lower -= bound_ext * span
        upper += bound_ext * span
        counts, edges = np.histogram(values, bins=bins, range=(lower, upper))
        widths = np.diff(edges)
        if np.any(widths <= 0.0):
            raise ValueError("El rango marginal no admite bins distintos en float64.")

        # Hay una pseudocuenta por bin y otra por cada cola; la masa total es 1.
        total = float(values.size + pseudocount * (bins + 2))
        masses = (counts.astype(np.float64) + pseudocount) / total
        tail_mass = float(pseudocount / total)
        cumulative_left = np.r_[0.0, np.cumsum(masses[:-1])] + tail_mass
        cumulative_right = (
            np.r_[np.cumsum(masses[:0:-1])[::-1], 0.0] + tail_mass
        )
        density = masses / widths
        # La densidad es continua en la unión entre soporte y cada cola.
        left_scale = tail_mass / density[0]
        right_scale = tail_mass / density[-1]
        return cls(
            edges=edges,
            masses=masses,
            cumulative_left=cumulative_left,
            cumulative_right=cumulative_right,
            log_density=np.log(density),
            tail_mass=tail_mass,
            left_scale=float(left_scale),
            right_scale=float(right_scale),
        )

    def transform(self, values):
        """Devuelve la gaussiana marginal y su logjacobiano por observación."""
        values = np.asarray(values, dtype=np.float64)
        gaussian = np.empty_like(values)
        log_pdf = np.empty_like(values)
        left = values < self.edges[0]
        right = values >= self.edges[-1]
        interior = ~(left | right)
        log_tail = np.log(self.tail_mass)

        if np.any(left):
            log_cdf = log_tail + (values[left] - self.edges[0]) / self.left_scale
            gaussian[left] = ndtri_exp(log_cdf)
            log_pdf[left] = log_cdf - np.log(self.left_scale)
        if np.any(right):
            log_survival = (
                log_tail - (values[right] - self.edges[-1]) / self.right_scale
            )
            gaussian[right] = -ndtri_exp(log_survival)
            log_pdf[right] = log_survival - np.log(self.right_scale)
        if np.any(interior):
            x = values[interior]
            index = np.searchsorted(self.edges, x, side="right") - 1
            width = self.edges[index + 1] - self.edges[index]
            position = (x - self.edges[index]) / width
            cdf = self.cumulative_left[index] + position * self.masses[index]
            survival = (
                self.cumulative_right[index]
                + (1.0 - position) * self.masses[index]
            )
            # Calcular ambas colas evita la cancelación numérica de 1 - CDF.
            lower_half = cdf <= 0.5
            z = np.empty_like(x)
            z[lower_half] = ndtri_exp(np.log(cdf[lower_half]))
            z[~lower_half] = -ndtri_exp(np.log(survival[~lower_half]))
            gaussian[interior] = z
            log_pdf[interior] = self.log_density[index]

        log_jacobian = log_pdf + 0.5 * (gaussian * gaussian + _LOG_2PI)
        return gaussian, log_jacobian


@dataclass
class _RBIGLayer:
    marginals: list[_MarginalHistogram]
    mean: np.ndarray
    rotation: np.ndarray


class RBIGDetector:
    """Detector RBIG; ``score_samples`` devuelve mayor valor para más anomalía.

    Parámetros admitidos en ``params``: ``n_layers`` (20), ``bins`` (128),
    ``pseudocount`` (0.5), ``bound_ext`` (0.3) y ``rotation`` (``"pca"``).
    Las capas y los bins limitan la complejidad del modelo, no las filas usadas.
    ``random_state`` se registra por uniformidad de la comparación; la rotación
    PCA completa mediante eigh es determinista y no emplea muestreo aleatorio.
    ``y`` se ignora, incluso cuando se facilita: este ajuste no filtra etiquetas.
    """

    def __init__(self, params=None, random_state=42, threads=2):
        self.params = {} if params is None else dict(params)
        self.random_state = random_state
        self.threads = threads

    @staticmethod
    def _array(X, n_features=None):
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("X debe ser una matriz con al menos una columna.")
        if n_features is not None and values.shape[1] != n_features:
            raise ValueError("X no tiene el número de columnas usado en fit.")
        if not np.isfinite(values).all():
            raise ValueError("X contiene valores no finitos.")
        return values

    def fit(self, X, y=None):
        """Ajusta todas las marginales y rotaciones usando únicamente X."""
        started = perf_counter()
        parameters = {
            "n_layers": 20,
            "bins": 128,
            "pseudocount": 0.5,
            "bound_ext": 0.3,
            "rotation": "pca",
        }
        unknown = set(self.params) - set(parameters)
        if unknown:
            raise ValueError(f"Parámetros RBIG no admitidos: {sorted(unknown)}")
        parameters.update(self.params)
        for name, minimum in (("n_layers", 1), ("bins", 2)):
            number = parameters[name]
            if (
                isinstance(number, (bool, np.bool_))
                or not isinstance(number, (int, np.integer))
                or number < minimum
            ):
                raise ValueError(f"{name} debe ser un entero >= {minimum}.")
            parameters[name] = int(number)
        for name, positive in (("pseudocount", True), ("bound_ext", False)):
            number = float(parameters[name])
            if not np.isfinite(number) or number < 0 or (positive and number == 0):
                raise ValueError(f"{name} debe ser finito y {'positivo' if positive else 'no negativo'}.")
            parameters[name] = number
        if str(parameters["rotation"]).lower() != "pca":
            raise ValueError("Esta adaptación implementa únicamente rotation='pca'.")
        parameters["rotation"] = "pca"
        if not isinstance(self.threads, (int, np.integer)) or self.threads < 1:
            raise ValueError("threads debe ser un entero positivo.")

        values = self._array(X)
        if values.shape[0] < 2:
            raise ValueError("RBIG necesita al menos dos filas de entrenamiento.")
        self.n_features_in_ = int(values.shape[1])
        self.layers_ = []
        current = values.copy()
        with threadpool_limits(limits=int(self.threads)):
            for _ in range(parameters["n_layers"]):
                marginals = []
                gaussian = np.empty_like(current)
                for column in range(self.n_features_in_):
                    marginal = _MarginalHistogram.fit(
                        current[:, column],
                        parameters["bins"],
                        parameters["pseudocount"],
                        parameters["bound_ext"],
                    )
                    gaussian[:, column], _ = marginal.transform(current[:, column])
                    marginals.append(marginal)
                mean = gaussian.mean(axis=0)
                centered = gaussian - mean
                covariance = centered.T @ centered / (len(centered) - 1)
                _, vectors = np.linalg.eigh(covariance)
                rotation = vectors[:, ::-1].copy()
                # Convención de signos fija; PCA conserva todas las dimensiones.
                pivots = np.argmax(np.abs(rotation), axis=0)
                signs = np.sign(rotation[pivots, np.arange(self.n_features_in_)])
                rotation *= signs
                current = centered @ rotation
                if not np.isfinite(current).all():
                    raise FloatingPointError("La transformación RBIG no es finita.")
                self.layers_.append(_RBIGLayer(marginals, mean, rotation))
        self.training_info_ = {
            "modelo": "RBIG adaptado: histograma + Gaussianización + PCA",
            "parametros": parameters,
            "filas_entrenamiento": int(values.shape[0]),
            "columnas": self.n_features_in_,
            "filas_por_capa": int(values.shape[0]),
            "submuestreo": False,
            "etiquetas_utilizadas": False,
            "semilla": None if self.random_state is None else int(self.random_state),
            "rotacion_determinista": True,
            "hilos": int(self.threads),
            "segundos_ajuste": float(perf_counter() - started),
            "fuente_url": _SOURCE_URL,
            "fuente_commit": "a9124fc07d8e851833802d604fe552b6327ab980",
            "referencia_miax": (
                "5_Inteligencia_Artificial_Generativa/2026_Anomalias/"
                "scripts_deep/AD_rbig_demo_clase.ipynb, celdas 2 y 6"
            ),
            "adaptaciones": [
                "Histograma de bins equidistantes y pseudocuentas normalizadas.",
                "Colas exponenciales con masa explícita y CDF inversa sin clipping.",
                "Logdensidad y jacobianos calculados sin exponentiar probabilidades.",
                "Número fijo de capas elegido en validación, sin parada por entropía.",
                "Marginales y rotaciones congeladas al predecir; score independiente del lote.",
            ],
        }
        return self

    def score_samples(self, X):
        """Devuelve menos logdensidad por fila; no actualiza ningún estimador."""
        if not hasattr(self, "training_info_"):
            raise RuntimeError("Debe ejecutarse fit antes de score_samples.")
        values = self._array(X, self.n_features_in_)
        current = values.copy()
        log_jacobian = np.zeros(len(current), dtype=np.float64)
        with threadpool_limits(limits=int(self.threads)):
            for layer in self.layers_:
                gaussian = np.empty_like(current)
                for column, marginal in enumerate(layer.marginals):
                    gaussian[:, column], contribution = marginal.transform(
                        current[:, column]
                    )
                    log_jacobian += contribution
                current = (gaussian - layer.mean) @ layer.rotation
        log_base_density = -0.5 * (
            np.sum(current * current, axis=1) + self.n_features_in_ * _LOG_2PI
        )
        scores = -(log_base_density + log_jacobian)
        if not np.isfinite(scores).all():
            raise FloatingPointError("La logdensidad RBIG no es finita.")
        return scores
