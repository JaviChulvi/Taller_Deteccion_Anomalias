"""Autoencoder tabular para comparar detección de anomalías sin etiquetas.

Origen docente: MIAX, ``2026_Anomalias/scripts_deep/AD_AE_simple_clase.ipynb``,
celdas 4, 8, 11 y 28 (numeración desde 1). Se conserva el principio de aprender
la reconstrucción y usar su error por fila como anomalía. Frente al ejemplo de
MNIST, esta adaptación usa PyTorch CPU, salida lineal y MSE para variables
numéricas estandarizadas; no utiliza sigmoide ni entropía cruzada de píxeles.

El harness debe ajustar StandardScaler exclusivamente con su partición de
entrenamiento. Este módulo no transforma datos ni recibe validación externa:
elige las épocas con una partición interna por grupos de filas idénticas y
vuelve a entrenar desde cero con TODAS las filas recibidas. ``y`` se ignora.
"""

from contextlib import contextmanager
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn


@contextmanager
def _cpu_determinista(threads):
    """Limita CPU y restaura la configuración global de PyTorch al terminar."""
    anteriores = torch.get_num_threads()
    determinismo = torch.are_deterministic_algorithms_enabled()
    solo_aviso = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.use_deterministic_algorithms(determinismo, warn_only=solo_aviso)
        torch.set_num_threads(anteriores)


class AutoencoderDetector:
    """AE denso con MSE por fila; un score mayor indica mayor anomalía.

    ``params`` admite ``bottleneck`` (4 u 8), ``hidden`` (32 o 64),
    ``weight_decay``, ``max_epochs`` (al menos 20), ``patience`` y ``batch_size``.
    El aprendizaje usa Adam con LR 1e-3 y lotes completos, incluido el último
    lote parcial. La selección de épocas reserva aproximadamente un 10 % de
    los grupos únicos internos, nunca fragmenta duplicados y solo considera
    candidatos a partir de la época 20. Después se reajusta sobre todas las
    filas durante el número de épocas seleccionado.

    ``training_info_`` contiene parámetros, tamaños e historiales serializables
    como JSON. El detector se puede guardar con joblib si este módulo es
    importable; ``model_.state_dict()`` también permite guardar pesos con torch.
    El determinismo se refiere al mismo entorno CPU y versiones de librerías.
    """

    _DEFAULTS = {
        "bottleneck": 4,
        "hidden": 32,
        "weight_decay": 1e-5,
        "max_epochs": 100,
        "patience": 10,
        "batch_size": 512,
    }
    _MIN_EPOCHS = 20
    _LEARNING_RATE = 1e-3
    _VALIDATION_FRACTION = 0.1

    def __init__(self, params=None, random_state=42, threads=2):
        if params is not None and not isinstance(params, dict):
            raise TypeError("params debe ser un diccionario o None.")
        desconocidos = set(params or {}) - self._DEFAULTS.keys()
        if desconocidos:
            raise ValueError(f"Parámetros desconocidos: {sorted(desconocidos)}")
        self.params = {**self._DEFAULTS, **(params or {})}
        for nombre in ("bottleneck", "hidden", "max_epochs", "patience", "batch_size"):
            valor = self.params[nombre]
            if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < 1:
                raise ValueError(f"{nombre} debe ser un entero positivo.")
            self.params[nombre] = int(valor)
        if self.params["bottleneck"] not in (4, 8):
            raise ValueError("bottleneck debe ser 4 u 8.")
        if self.params["hidden"] not in (32, 64):
            raise ValueError("hidden debe ser 32 o 64.")
        if self.params["max_epochs"] < self._MIN_EPOCHS:
            raise ValueError("max_epochs debe ser al menos 20.")
        decaimiento = self.params["weight_decay"]
        if (
            isinstance(decaimiento, bool)
            or not isinstance(decaimiento, Real)
            or not np.isfinite(decaimiento)
            or decaimiento < 0
        ):
            raise ValueError("weight_decay debe ser finito y no negativo.")
        self.params["weight_decay"] = float(decaimiento)
        for nombre, valor in (("random_state", random_state), ("threads", threads)):
            minimo = 0 if nombre == "random_state" else 1
            if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < minimo:
                raise ValueError(f"{nombre} debe ser un entero >= {minimo}.")
        self.random_state = int(random_state)
        self.threads = int(threads)

    @staticmethod
    def _validar_X(X):
        datos = np.asarray(X)
        if datos.ndim != 2 or datos.shape[1] == 0:
            raise ValueError("X debe ser una matriz numérica con al menos una columna.")
        if not np.issubdtype(datos.dtype, np.number) or np.iscomplexobj(datos):
            raise ValueError("X debe contener números reales.")
        if not np.isfinite(datos).all():
            raise ValueError("X contiene valores no finitos.")
        return datos

    @staticmethod
    def _tensor(datos):
        with np.errstate(over="raise", invalid="raise"):
            matriz = np.array(datos, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(matriz).all():
            raise ValueError("X no puede representarse mediante float32 finitos.")
        return torch.from_numpy(matriz)

    def _crear_modelo(self):
        torch.manual_seed(self.random_state)
        ancho = self.params["hidden"]
        cuello = self.params["bottleneck"]
        return nn.Sequential(
            nn.Linear(self.n_features_in_, ancho),
            nn.ReLU(),
            nn.Linear(ancho, cuello),
            nn.ReLU(),
            nn.Linear(cuello, ancho),
            nn.ReLU(),
            nn.Linear(ancho, self.n_features_in_),
        ).cpu()

    def _scores_tensor(self, modelo, datos):
        """Calcula scores por lotes, sin construir gradientes ni omitir filas."""
        modelo.eval()
        resultado = np.empty(len(datos), dtype=np.float64)
        tamano = self.params["batch_size"]
        with torch.inference_mode():
            for inicio in range(0, len(datos), tamano):
                lote = datos[inicio : inicio + tamano]
                error = (modelo(lote) - lote).square().mean(dim=1)
                if not torch.isfinite(error).all().item():
                    raise FloatingPointError("El autoencoder produjo scores no finitos.")
                resultado[inicio : inicio + len(lote)] = error.numpy()
        return resultado

    def _entrenar(self, datos, epocas, validacion=None):
        modelo = self._crear_modelo()
        optimizador = torch.optim.Adam(
            modelo.parameters(),
            lr=self._LEARNING_RATE,
            weight_decay=self.params["weight_decay"],
        )
        generador = torch.Generator(device="cpu").manual_seed(self.random_state)
        historial = []
        mejor_epoca = epocas
        mejor_error = float("inf")
        tamano = self.params["batch_size"]
        for epoca in range(1, epocas + 1):
            modelo.train()
            orden = torch.randperm(len(datos), generator=generador)
            suma = 0.0
            for inicio in range(0, len(datos), tamano):
                lote = datos[orden[inicio : inicio + tamano]]
                optimizador.zero_grad(set_to_none=True)
                perdida = nn.functional.mse_loss(modelo(lote), lote)
                if not torch.isfinite(perdida).item():
                    raise FloatingPointError("El entrenamiento produjo una pérdida no finita.")
                perdida.backward()
                optimizador.step()
                suma += float(perdida.detach().item()) * len(lote)
            registro = {"epoch": epoca, "train_mse": suma / len(datos)}
            if validacion is not None:
                error = float(self._scores_tensor(modelo, validacion).mean())
                registro["validation_mse"] = error
                if epoca >= self._MIN_EPOCHS and error < mejor_error:
                    mejor_error = error
                    mejor_epoca = epoca
            historial.append(registro)
            if (
                validacion is not None
                and epoca >= self._MIN_EPOCHS
                and epoca - mejor_epoca >= self.params["patience"]
            ):
                break
        modelo.eval()
        return modelo, mejor_epoca, historial

    def fit(self, X, y=None):
        """Selecciona épocas internamente y reajusta todas las filas, sin usar y."""
        datos = self._validar_X(X)
        if len(datos) < 2:
            raise ValueError("Se necesitan al menos dos filas para la validación interna.")
        # Los grupos se calculan ANTES de float32 para respetar las filas originales.
        unicos, grupos = np.unique(datos, axis=0, return_inverse=True)
        n_grupos = len(unicos)
        del unicos
        if n_grupos < 2:
            raise ValueError("Se necesitan al menos dos grupos de filas diferentes.")
        n_validacion = max(1, int(round(n_grupos * self._VALIDATION_FRACTION)))
        rng = np.random.default_rng(self.random_state)
        grupos_validacion = rng.permutation(n_grupos)[:n_validacion]
        mascara = np.isin(grupos, grupos_validacion)
        self.n_features_in_ = int(datos.shape[1])
        tensor = self._tensor(datos)
        indices_train = torch.from_numpy(np.flatnonzero(~mascara))
        indices_val = torch.from_numpy(np.flatnonzero(mascara))
        with _cpu_determinista(self.threads):
            _, mejor_epoca, seleccion = self._entrenar(
                tensor[indices_train],
                self.params["max_epochs"],
                validacion=tensor[indices_val],
            )
            # Nueva inicialización: la validación interna vuelve al entrenamiento.
            modelo, _, reajuste = self._entrenar(tensor, mejor_epoca)
            self._scores_tensor(modelo, tensor)
        self.model_ = modelo
        self.training_info_ = {
            "model": "autoencoder_denso",
            "torch_version": str(torch.__version__),
            "device": "cpu",
            "threads": self.threads,
            "random_state": self.random_state,
            "params": dict(self.params),
            "learning_rate": self._LEARNING_RATE,
            "min_epochs": self._MIN_EPOCHS,
            "fit_rows": int(len(datos)),
            "features": self.n_features_in_,
            "unique_row_groups": n_grupos,
            "validation_strategy": "grupos internos de filas idénticas",
            "internal_train_rows": int((~mascara).sum()),
            "internal_validation_rows": int(mascara.sum()),
            "internal_train_groups": int(n_grupos - n_validacion),
            "internal_validation_groups": n_validacion,
            "group_overlap": 0,
            "selected_epochs": mejor_epoca,
            "selection_epochs_run": len(seleccion),
            "refit_rows": int(len(datos)),
            "refit_epochs": len(reajuste),
            "labels_used": False,
            "score": "MSE por fila; mayor = mayor anomalía",
            "selection_history": seleccion,
            "refit_history": reajuste,
        }
        return self

    def score_samples(self, X):
        """Devuelve un vector finito con MSE por fila; mayor significa anomalía."""
        if not hasattr(self, "model_"):
            raise RuntimeError("El detector debe ajustarse mediante fit antes de puntuar.")
        datos = self._validar_X(X)
        if datos.shape[1] != self.n_features_in_:
            raise ValueError(f"Se esperaban {self.n_features_in_} columnas.")
        with _cpu_determinista(self.threads):
            return self._scores_tensor(self.model_, self._tensor(datos))
