"""Estudio descriptivo completo de los ZIP originales, sin modificarlos.

Desde la raíz del repositorio: python javi/estudio_datos/perfilar_datos.py
Las estadísticas utilizan todas las filas. No se entrena un detector de fraude.
"""
from pathlib import Path
import hashlib
import json
import platform
import zipfile
import importlib.metadata

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SALIDA = Path(__file__).resolve().parent
RAIZ = SALIDA.parent.parent
FIGURAS = SALIDA / "figuras"
FIGURAS.mkdir(exist_ok=True)
VARIABLES = [f"X{i}" for i in range(1, 21)] + ["Amount"]
QUANTILES = [.001, .01, .05, .25, .5, .75, .95, .99, .999]
datos = {n: pd.read_csv(RAIZ / f"credit_card_{n}.csv.zip")
         for n in ["construction", "production"]}
const, prod = datos.values()


def guardar_json(nombre, contenido):
    def convertir(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    (SALIDA / nombre).write_text(json.dumps(contenido, ensure_ascii=False,
        indent=2, default=convertir, allow_nan=False), encoding="utf-8")


def perfil(serie):
    x = serie.dropna()
    q = x.quantile(QUANTILES)
    iqr = q.loc[.75] - q.loc[.25]
    return {"n": len(serie), "n_validos": len(x), "nulos": int(serie.isna().sum()),
            "tipo": str(serie.dtype), "distintos": int(x.nunique()),
            "infinitos": int(np.isinf(x).sum()), "ceros": int((x == 0).sum()),
            "negativos": int((x < 0).sum()), "minimo": float(x.min()),
            "media": float(x.mean()), "desviacion": float(x.std()),
            "maximo": float(x.max()), "asimetria": float(x.skew()),
            "curtosis_exceso": float(x.kurt()),
            "cuantiles": {str(k): float(v) for k, v in q.items()},
            "fuera_tukey_1_5_iqr": int(((x < q.loc[.25] - 1.5 * iqr) |
                                             (x > q.loc[.75] + 1.5 * iqr)).sum()),
            "fuera_tukey_3_iqr": int(((x < q.loc[.25] - 3 * iqr) |
                                           (x > q.loc[.75] + 3 * iqr)).sum())}


manifiesto = []
for p in sorted(RAIZ.iterdir()):
    if not p.is_file():
        continue
    fila = {"archivo": p.name, "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "papel": "metadatos Finder, sin datos analíticos" if p.name == ".DS_Store" else "fuente"}
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            fila["integridad_crc"] = z.testzip() is None
            fila["miembros"] = [{"nombre": i.filename, "bytes": i.file_size}
                                 for i in z.infolist()]
    manifiesto.append(fila)
guardar_json("manifiesto_fuentes.json", manifiesto)

resumen = {"fecha_estudio": "2026-10-19", "filas_total": len(const) + len(prod),
           "versiones": {m: importlib.metadata.version(m) for m in
                         ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib"]},
           "python": platform.python_version(), "conjuntos": {}}
perfiles = []
for n, d in datos.items():
    resumen["conjuntos"][n] = {"filas": len(d), "columnas": len(d.columns),
        "predictores": len(VARIABLES), "nulos_predictores": int(d[VARIABLES].isna().sum().sum()),
        "infinitos_predictores": int(np.isinf(d[VARIABLES]).sum().sum()),
        "nulos_class": int(d.Class.isna().sum()),
        "clases": {str(k): int(v) for k, v in d.Class.value_counts().items()},
        "variables_constantes": [c for c in VARIABLES if d[c].nunique() <= 1],
        "bytes_en_memoria": int(d.memory_usage(deep=True).sum()),
        "perfil": {c: perfil(d[c]) for c in VARIABLES},
        "amount_total": float(d.Amount.sum()),
        "amount_con_decimales_superiores_a_centimos": int((abs(d.Amount * 100 - (d.Amount * 100).round()) > 1e-7).sum())}
    for c, valores in resumen["conjuntos"][n]["perfil"].items():
        perfiles.append({"conjunto": n, "variable": c, **{k: v for k, v in valores.items() if k != "cuantiles"},
                         **{f"q_{k}": v for k, v in valores["cuantiles"].items()}})
resumen["prevalencia"] = float(const.Class.mean())
resumen["ratio_normales_por_fraude"] = int((const.Class == 0).sum()) / int(const.Class.sum())
resumen["accuracy_todo_normal"] = 1 - resumen["prevalencia"]

# Descripción por clase: separación marginal, no validación predictiva.
clases = {}
senales = []
y = const.Class.to_numpy()
for clase, d in const.groupby("Class"):
    clases[str(clase)] = {"filas": len(d), "amount_total": float(d.Amount.sum()),
                         "perfil": {c: perfil(d[c]) for c in VARIABLES}}
for c in VARIABLES:
    a = const.loc[const.Class == 0, c]
    b = const.loc[const.Class == 1, c]
    ks = stats.ks_2samp(a, b, method="asymp")
    auc = roc_auc_score(y, const[c])
    ap_mas = average_precision_score(y, const[c])
    ap_menos = average_precision_score(y, -const[c])
    senales.append({"variable": c, "media_normal": a.mean(), "media_fraude": b.mean(),
        "mediana_normal": a.median(), "mediana_fraude": b.median(),
        "ks_clases": ks.statistic, "auc_mayor_valor": auc,
        "ap_mayor_valor": ap_mas, "ap_menor_valor": ap_menos,
        "mejor_ap_descriptiva": max(ap_mas, ap_menos),
        "direccion_descriptiva": "mayor" if ap_mas >= ap_menos else "menor",
        "diferencia_medias_en_sd_normal": (b.mean() - a.mean()) / a.std()})
senales = sorted(senales, key=lambda s: s["mejor_ap_descriptiva"], reverse=True)

# Deriva marginal con tamaño de efecto y corrección por múltiples comparaciones.
deriva = []
for c in VARIABLES:
    a, b = const[c].to_numpy(), prod[c].to_numpy()
    ks = stats.ks_2samp(a, b, method="asymp")
    cortes = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    ca, cb = np.histogram(a, cortes)[0], np.histogram(b, cortes)[0]
    pa, pb = (ca + .5) / (len(a) + .5 * len(ca)), (cb + .5) / (len(b) + .5 * len(cb))
    deriva.append({"variable": c, "ks": ks.statistic, "p_ks": ks.pvalue,
        "psi_decil_suavizado": float(np.sum((pb - pa) * np.log(pb / pa))),
        "diferencia_medias_en_sd_construccion": (b.mean() - a.mean()) / a.std(ddof=1),
        "wasserstein_en_iqr_construccion": stats.wasserstein_distance(a, b) / (np.quantile(a, .75) - np.quantile(a, .25)),
        "produccion_fuera_rango_const": int(((b < a.min()) | (b > a.max())).sum()),
        "ratio_desviaciones": b.std(ddof=1) / a.std(ddof=1)})
orden = np.argsort([r["p_ks"] for r in deriva])
ajustados = np.minimum.accumulate(np.array([deriva[i]["p_ks"] for i in orden])[::-1] * len(orden) / np.arange(len(orden), 0, -1))[::-1]
for i, p in zip(orden, ajustados):
    deriva[i]["p_ks_bh"] = float(min(p, 1))

# Correlaciones completas y dimensionalidad lineal de predictores estandarizados.
correlaciones = {}
for etiqueta, d in [("construccion", const), ("produccion", prod),
                    ("normal", const[const.Class == 0]), ("fraude", const[const.Class == 1])]:
    corr = d[VARIABLES].corr()
    pares = sorted([{"a": VARIABLES[i], "b": VARIABLES[j], "r": corr.iloc[i, j]}
                    for i in range(len(VARIABLES)) for j in range(i)], key=lambda p: abs(p["r"]), reverse=True)
    valores = np.linalg.eigvalsh(corr.to_numpy())[::-1]
    correlaciones[etiqueta] = {"pearson": corr.to_dict(), "spearman": d[VARIABLES].corr(method="spearman").to_dict(),
        "pares_principales": pares[:20], "autovalores_correlacion": valores.tolist(),
        "componentes_95_pct_varianza_estandarizada": int(np.searchsorted(np.cumsum(valores) / valores.sum(), .95) + 1),
        "condicion_correlacion": float(valores.max() / valores.min())}

# Importes y extremos: reglas descriptivas, nunca etiquetas automáticas de fraude.
bordes = [-.01, 0, 1, 10, 50, 100, 500, 1000, 5000, np.inf]
tramos = []
for intervalo, d in const.groupby(pd.cut(const.Amount, bordes), observed=True):
    tramos.append({"tramo": str(intervalo), "filas": len(d), "fraudes": int(d.Class.sum()),
                   "tasa_fraude": float(d.Class.mean()), "importe_total": float(d.Amount.sum())})
extremos = []
for q in [.95, .99, .999]:
    limite = float(const.Amount.quantile(q))
    mascara = const.Amount > limite
    extremos.append({"cuantil": q, "limite": limite, "filas": int(mascara.sum()),
                     "fraudes": int(const.loc[mascara, "Class"].sum()),
                     "tasa_fraude": float(const.loc[mascara, "Class"].mean())})
q1, q3 = const[VARIABLES].quantile(.25), const[VARIABLES].quantile(.75)
raros = ((const[VARIABLES] < q1 - 1.5 * (q3 - q1)) | (const[VARIABLES] > q3 + 1.5 * (q3 - q1))).any(axis=1)
extremos_resumen = {"tukey_cualquier_variable": {"filas": int(raros.sum()), "fraudes": int(const.loc[raros, "Class"].sum()),
                    "precision": float(const.loc[raros, "Class"].mean()), "recall": float(const.loc[raros, "Class"].sum() / const.Class.sum())},
                    "colas_amount": extremos}

# Orden físico del archivo: sin afirmar que representa fechas o clientes.
bloques = []
for nombre, d in datos.items():
    for j, indices in enumerate(np.array_split(np.arange(len(d)), 10), 1):
        b = d.iloc[indices]
        bloques.append({"conjunto": nombre, "decil_posicion": j, "filas": len(b),
                        "fraudes": int(b.Class.sum()) if nombre == "construction" else None,
                        "amount_mediana": float(b.Amount.median()), "amount_media": float(b.Amount.mean()),
                        "medias_x": {c: float(b[c].mean()) for c in VARIABLES[:-1]}})

guardar_json("resumen_datos.json", resumen)
guardar_json("perfiles_por_clase.json", clases)
guardar_json("senales_univariantes.json", senales)
guardar_json("comparacion_distribuciones.json", deriva)
guardar_json("correlaciones.json", correlaciones)
guardar_json("importes_extremos.json", {"tramos": tramos, **extremos_resumen})
guardar_json("orden_filas.json", bloques)
guardar_json("perfiles_columnas.json", perfiles)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
azul, rojo, verde = "#265d8f", "#bc413b", "#39816c"


def guardar_figura(nombre):
    plt.savefig(FIGURAS / nombre, dpi=150, bbox_inches="tight")
    plt.close()


fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
ax[0].bar(["Normales", "Fraudes"], [(y == 0).sum(), (y == 1).sum()], color=[azul, rojo])
ax[0].set_yscale("log"); ax[0].set_ylabel("Transacciones (escala logarítmica)")
ax[0].set_title("Construcción: 295 fraudes entre 170.884 filas")
for clase, color in [(0, azul), (1, rojo)]:
    ax[1].hist(np.log1p(const.loc[const.Class == clase, "Amount"]), bins=45, density=True,
               histtype="step", linewidth=1.7, label="Normal" if clase == 0 else "Fraude", color=color)
ax[1].set(xlabel="log(1 + Amount), unidad monetaria sin documentar", ylabel="Densidad",
          title="Importes por clase: distribuciones normalizadas")
ax[1].legend(); fig.tight_layout(); guardar_figura("clases_importes.png")

fig, ejes = plt.subplots(7, 3, figsize=(15, 24))
for c, ax in zip(VARIABLES, ejes.flat):
    v = const[c]
    lim = np.quantile(v, [.001, .999])
    bins = np.linspace(lim[0], lim[1], 60)
    for clase, color in [(0, azul), (1, rojo)]:
        x = const.loc[const.Class == clase, c]
        cuentas, _ = np.histogram(x, bins)
        ax.stairs(cuentas / len(x) / np.diff(bins), bins, color=color, linewidth=1.4,
                   label="Normal" if clase == 0 else "Fraude")
    ax.set(title=c, ylabel="Densidad", xlim=lim)
ejes.flat[0].legend(); fig.suptitle("Todos los predictores por clase\nVista entre p0,1 y p99,9 de construcción; las colas excluidas no se eliminan del análisis", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, .96]); guardar_figura("distribuciones_por_clase.png")

fig, ejes = plt.subplots(7, 3, figsize=(15, 24))
for c, ax in zip(VARIABLES, ejes.flat):
    lim = np.quantile(const[c], [.001, .999]); bins = np.linspace(*lim, 60)
    for nombre, color in [("construction", azul), ("production", verde)]:
        x = datos[nombre][c]; cuentas, _ = np.histogram(x, bins)
        ax.stairs(cuentas / len(x) / np.diff(bins), bins, color=color, linewidth=1.3,
                  label="Construcción" if nombre == "construction" else "Producción")
    ax.set(title=c, ylabel="Densidad", xlim=lim)
ejes.flat[0].legend(); fig.suptitle("Construcción frente a producción: 21 predictores\nVista entre p0,1 y p99,9 de construcción; sin etiquetas de producción", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, .96]); guardar_figura("distribuciones_conjuntos.png")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(const[VARIABLES].corr(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(21), VARIABLES, rotation=90); ax.set_yticks(range(21), VARIABLES)
ax.set_title("Correlaciones de Pearson en construcción")
fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout(); guardar_figura("correlaciones.png")

fig, ejes = plt.subplots(1, 2, figsize=(13, 7))
top = senales[:12][::-1]
ejes[0].barh([s["variable"] for s in top], [s["mejor_ap_descriptiva"] for s in top], color=azul)
ejes[0].set(title="Separación univariante en construcción", xlabel="AP descriptiva: mejor signo elegido sobre los mismos datos")
ejes[0].axvline(resumen["prevalencia"], color=rojo, linestyle="--", label="Prevalencia")
ejes[0].legend()
dd = sorted(deriva, key=lambda r: r["ks"])
ejes[1].barh([s["variable"] for s in dd], [s["ks"] for s in dd], color=verde)
ejes[1].set(title="Distancia marginal entre conjuntos", xlabel="Estadístico KS (máxima diferencia de distribuciones acumuladas)")
fig.tight_layout(); guardar_figura("senal_y_deriva.png")

print(json.dumps({"resumen": {n: {k: v for k, v in r.items() if k != "perfil"} for n, r in resumen["conjuntos"].items()},
                  "senales_principales": senales[:7], "deriva_maxima": sorted(deriva, key=lambda r: r["ks"], reverse=True)[:4],
                  "pares_correlacion": correlaciones["construccion"]["pares_principales"][:5],
                  "extremos": extremos_resumen}, indent=2, ensure_ascii=False, default=lambda x: x.item()), flush=True)
