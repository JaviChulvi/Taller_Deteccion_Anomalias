"""Genera un informe HTML autónomo a partir de las evidencias del estudio."""
from pathlib import Path
from html import escape
import base64
import json

BASE = Path(__file__).resolve().parent


def leer(nombre):
    return json.loads((BASE / nombre).read_text(encoding="utf-8"))


def numero(x, dec=2):
    if x is None:
        return "No disponible"
    return f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def entero(x):
    return numero(x, 0)


def porcentaje(x, dec=3):
    return numero(100 * x, dec) + " %"


def tabla(cabecera, filas):
    return '<div class="tabla"><table><thead><tr>' + "".join(f"<th>{escape(str(x))}</th>" for x in cabecera) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in fila) + "</tr>" for fila in filas) + "</tbody></table></div>"


def figura(nombre, pie):
    datos = base64.b64encode((BASE / "figuras" / nombre).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{datos}" alt="{escape(pie)}"><figcaption>{escape(pie)}</figcaption></figure>'


res = leer("resumen_datos.json")
clases = leer("perfiles_por_clase.json")
senales = leer("senales_univariantes.json")
drift = leer("comparacion_distribuciones.json")
corr = leer("correlaciones.json")
importe = leer("importes_extremos.json")
duplicados = leer("duplicados_validacion.json")
nb = leer("auditoria_notebook.json")
manifest = leer("manifiesto_fuentes.json")
multi = leer("deriva_multivariante.json")
exactos = duplicados["analisis"]["X1_X20_Amount"]
reducidos = duplicados["analisis"]["X1_X20"]
pc = res["conjuntos"]["construction"]["perfil"]
pp = res["conjuntos"]["production"]["perfil"]
variables = list(pc)
secciones = []


def seccion(id_, titulo, contenido):
    secciones.append((id_, titulo, f'<section id="{id_}"><h2>{titulo}</h2>{contenido}</section>'))


seccion("lectura", "1. Qué tenemos y qué implica", f"""
<p>Se han examinado íntegramente los cuatro archivos de trabajo y las <strong>{entero(res['filas_total'])} filas</strong> disponibles.
El conjunto permite estudiar detección de fraude y construir un ranking: hay variables con separación marcada entre clases y etiquetas utilizables en construcción.
La dificultad principal está en evaluar con una minoría de solo <strong>295 fraudes</strong> y con patrones repetidos entre particiones.</p>
<div class="hallazgo"><strong>Prioridad antes de modelar:</strong> fijar una validación que controle las repeticiones, medir Average Precision con el sentido correcto del score y conservar intacto el orden de producción.</div>
{tabla(['Propiedad', 'Construcción', 'Producción'], [
['Transacciones', '170.884', '113.923'], ['Variables predictoras', '21', '21'],
['Columnas totales', '22', '22'], ['Fraudes observados', '295', 'Desconocido'],
['No fraudes observados', '170.589', 'Desconocido'], ['Prevalencia observada', porcentaje(res['prevalencia'], 6), 'No estimable con etiquetas locales'],
['Nulos en predictores', '0', '0'], ['Infinitos en predictores', '0', '0'],
['Nulos en Class', '0', '113.923 (100 %)'],
['Filas repetidas adicionales sobre 21 predictores', '4.167', '2.109']])}
<p>Se han calculado perfiles completos, cuantiles, colas, ceros, cardinalidades, correlaciones, separación por clase, duplicados, coincidencias entre archivos,
solape del split y diferencias de distribución. Los resultados nuevos son descriptivos y diagnósticos. El modelo de fraude del notebook no se ha reentrenado,
no hay una AP validada de ese modelo y no se ha producido el CSV de entrega.</p>
""")

filas_ficheros = []
for f in manifest:
    nombre = f["archivo"]
    if nombre.endswith(".zip"):
        rol = "Datos etiquetados" if "construction" in nombre else "Datos de inferencia con Class vacía"
        detalle = ", ".join(i["nombre"] + " (" + entero(i["bytes"]) + " bytes)" for i in f["miembros"])
    elif nombre.endswith(".pdf"):
        rol, detalle = "Enunciado", "1 página, texto completo y revisión visual"
    elif nombre.endswith(".ipynb"):
        rol, detalle = "Notebook inicial", "25 celdas: 3 markdown, 21 de código no vacías y 1 de código vacía"
    else:
        rol, detalle = "Metadatos de macOS", "No contiene material analítico del taller"
    filas_ficheros.append([nombre, entero(f["bytes"]), rol, detalle])
seccion("inventario", "2. Inventario y procedencia", tabla(["Archivo", "Bytes", "Función", "Contenido"], filas_ficheros) + """
<p>Los dos ZIP superan la comprobación CRC y contienen exactamente un CSV cada uno. Se leen directamente desde el ZIP: no hace falta descargar ni extraer de nuevo.
El directorio original no contiene README, diccionario de variables, requirements.txt, pyproject.toml, lockfile, modelos serializados ni CSV de scores.
No hay repositorio Git ni instrucciones AGENTS.md locales en este directorio.</p>
<p>Las fuentes locales no documentan el origen exacto, la licencia, el periodo, los identificadores de transacción o cliente,
la moneda ni la transformación que produjo X1–X20. No se atribuye automáticamente este conjunto a un dataset público por su tamaño o semejanza.
Tampoco se han descargado datos adicionales ni buscado las etiquetas ocultas de producción.</p>
<details><summary>Huellas SHA-256 de los archivos de entrada</summary>""" +
tabla(["Archivo", "SHA-256"], [[f["archivo"], f["sha256"]] for f in manifest]) + "</details>")

seccion("requisitos", "3. Requisitos contrastados", """
<p>El PDF y el texto inicial del notebook se complementan. La siguiente tabla distingue el requisito explícito de las decisiones metodológicas que habrá que tomar.</p>
""" + tabla(["Aspecto", "Requisito y fuente", "Estado actual"], [
["Trabajo", "En grupos y en Python (PDF)", "Faltan integrantes en el notebook"],
["Fecha", "29 de octubre, antes de las 18:30, aula virtual (PDF)", "El PDF no imprime año ni zona horaria"],
["Notebook", "Integrantes, código, justificación y reflexión final (PDF)", "Plantilla parcial, sin justificación ni reflexión final"],
["CSV", "Una columna y un score por fila de producción (PDF y celda 2)", "Pendiente: 113.923 scores en el orden original"],
["Puntuación", "Resultados 75 %; notebook 25 % (PDF)", "No se detalla conversión del resultado en nota"],
["Métrica", "AUPR; se señala average_precision_score (celda 2)", "Importada, nunca calculada"],
["Sentido del score", "A mayor score, mayor sospecha de fraude (celda 2)", "La log-densidad actual tiene el sentido contrario"],
["Formato pendiente", "No se fija cabecera, nombre de columna ni nombre de archivo", "Comprobar convención al preparar entrega"],
["Modelos y validación", "No se impone algoritmo ni split en el PDF", "GMM y 70/30 son elecciones del ejemplo"]]) + """
<p><strong>Average Precision (AP)</strong> es la implementación concreta que señala el notebook. No debe sustituirse sin justificación por integración trapezoidal de una curva PR:
son cálculos diferentes. Para AP basta un score que ordene el riesgo; no se exige probabilidad calibrada ni una decisión binaria.
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html">Documentación de average_precision_score</a>.</p>
""")

seccion("esquema", "4. Esquema, calidad y precisión numérica", """
<p>Cada fila representa una transacción según la descripción del ejercicio. Los CSV comparten exactamente el mismo orden de columnas:
X1…X20, Amount y Class. X1…X20 y Amount se leen como float64. Class es int64 en construcción y float64 completamente nulo en producción.</p>
""" + tabla(["Campo", "Interpretación respaldada", "Tratamiento recomendado"], [
["X1–X20", "Variables numéricas sin semántica documentada", "Conservar nombres y orden; no inventar significado"],
["Amount", "Importe de la transacción, moneda no documentada", "Evaluar escala original y log1p dentro de validación"],
["Class en construcción", "0: no fraude; 1: fraude", "Usar como objetivo, excluir siempre de X"],
["Class en producción", "Etiqueta ausente", "Excluir; no imputar ceros ni usarla como predictor"],
["Número de fila", "Posición física dentro del archivo", "Conservar para alinear salida; no tratar como fecha ni variable predictora"]]) + """
<p>No hay valores ausentes ni infinitos en ninguno de los 21 predictores, ni variables constantes, ni Amount negativo. La falta de Class en producción es esperada y no requiere imputación.
No hay categorías, fechas o identificadores explícitos que permitan validar por cliente o por tiempo.</p>
<p><strong>Cuantización:</strong> X1–X20 son compatibles con múltiplos de 2 salvo residuos de coma flotante.
El máximo error de X/2 respecto al entero más cercano es aproximadamente 4,55×10⁻¹³.
Amount es compatible con dos decimales; el error máximo de 100·Amount frente al entero es aproximadamente 2,33×10⁻¹⁰.
Esto reduce la resolución observable y ayuda a interpretar las coincidencias. No identifica la transformación original.
La sensibilidad que normaliza solo esos residuos no cambia los recuentos internos de duplicados.</p>
<p>Hay valores cero legítimos desde el punto de vista del archivo. No se dispone de una regla de negocio que permita descartarlos.
Las desviaciones de los predictores de construcción van de 152,47 en X20 a 390,92 en X1; Amount tiene 363,49.
La escala merece evaluación para métodos sensibles a distancia, pero su diferencia no demuestra por sí sola un error del GMM.</p>
""")

def tabla_perfiles(perfiles):
    return tabla(["Variable", "Distintos", "Mínimo", "p1", "p25", "Mediana", "p75", "p99", "Máximo", "Media", "Desv."],
      [[c, entero(p["distintos"]), numero(p["minimo"]), numero(p["cuantiles"]["0.01"]), numero(p["cuantiles"]["0.25"]),
        numero(p["cuantiles"]["0.5"]), numero(p["cuantiles"]["0.75"]), numero(p["cuantiles"]["0.99"]),
        numero(p["maximo"]), numero(p["media"]), numero(p["desviacion"])] for c, p in perfiles.items()])

seccion("variables", "5. Perfil de las 21 variables", """
<p>Las estadísticas de esta sección usan todas las filas, sin eliminar duplicados ni recortar extremos. Los valores mostrados se redondean a dos decimales;
los JSON conservan mayor precisión. Los percentiles p0,1, p5, p95 y p99,9 también están en los anexos de resultados.</p>
<h3>Construcción</h3>""" + tabla_perfiles(pc) + "<h3>Producción</h3>" + tabla_perfiles(pp) +
"<h3>Ceros y forma de las distribuciones en construcción</h3>" +
tabla(["Variable", "Ceros", "Negativos", "Asimetría", "Curtosis exceso", "Fuera 1,5 IQR", "% filas"],
[[c, entero(p["ceros"]), entero(p["negativos"]), numero(p["asimetria"]), numero(p["curtosis_exceso"]),
entero(p["fuera_tukey_1_5_iqr"]), porcentaje(p["fuera_tukey_1_5_iqr"] / p["n"])] for c,p in pc.items()]) + """
<p>La regla de Tukey marca valores por debajo de Q1−1,5·IQR o por encima de Q3+1,5·IQR. Es una descripción de colas,
no una prueba de error de datos ni una etiqueta de fraude. La asimetría y curtosis son sensibles a los extremos y se interpretan junto con los cuantiles.</p>
""")

seccion("clases", "6. Desbalance y separación entre clases", f"""
<p>Hay {numero(res['ratio_normales_por_fraude'], 1)} transacciones normales por cada fraude. Un sistema que siempre predijera normal tendría
<strong>{porcentaje(res['accuracy_todo_normal'], 4)} de accuracy</strong> y detectaría cero fraudes.
Un score constante tiene AP igual a la prevalencia: <strong>{numero(res['prevalencia'], 6)}</strong>. Estas referencias explican por qué el porcentaje de aciertos no debe guiar el taller.</p>
{figura('clases_importes.png', 'Los histogramas se normalizan por clase; la altura no representa el número absoluto de fraudes.')}
<p>Para cada predictor se ha calculado AP usando el valor y su negativo, y se muestra el mejor de los dos como medida descriptiva.
<strong>El signo y el ranking de variables se han elegido sobre toda construcción: estos números no son una evaluación fuera de muestra ni una estimación de AP de producción.</strong>
Sirven para localizar señal y proponer experimentos posteriores.</p>
""" + tabla(["Variable", "Sentido asociado", "AP descriptiva", "KS clases", "Mediana normal", "Mediana fraude", "Δ media / SD normal"],
[[s["variable"], s["direccion_descriptiva"], numero(s["mejor_ap_descriptiva"], 6), numero(s["ks_clases"], 4),
numero(s["mediana_normal"]), numero(s["mediana_fraude"]), numero(s["diferencia_medias_en_sd_normal"], 3)] for s in senales]) + """
<p>X17, X14, X12 y X10 muestran mayor sospecha en valores bajos, y X11 en valores altos. X13 y X15 presentan bastante más solapamiento marginal.
Una variable débil por sí sola aún puede aportar información en combinación con otras; este análisis no justifica eliminarla automáticamente.</p>
""" + figura("senal_y_deriva.png", "Izquierda: señal marginal descriptiva. Derecha: diferencias marginales entre construcción y producción; son magnitudes distintas.") +
"<details><summary>Distribuciones de las 21 variables por clase</summary>" + figura("distribuciones_por_clase.png", "Las vistas se limitan a p0,1–p99,9 de construcción para facilitar la lectura. Las estadísticas y el análisis conservan todas las colas.") + "</details>" +
"<details><summary>Estadísticos completos de cada clase</summary><h3>No fraude</h3>" + tabla_perfiles(clases["0"]["perfil"]) + "<h3>Fraude</h3>" + tabla_perfiles(clases["1"]["perfil"]) + "</details>")

seccion("importes", "7. Importes y extremos", """
<p>Amount es muy asimétrico y tiene una cola larga. Su media en construcción es 132,12 y su mediana 33,00.
Los fraudes tienen media 179,25 pero mediana 16,50; las transacciones normales tienen mediana 33,03.
Por tanto, ni importe alto ni importe bajo constituyen por sí solos una regla suficiente.</p>
""" + tabla(["Medida de Amount", "Normal", "Fraude", "Producción sin etiqueta"], [
["Media", numero(clases["0"]["perfil"]["Amount"]["media"]), numero(clases["1"]["perfil"]["Amount"]["media"]), numero(pp["Amount"]["media"])],
["Mediana", "33,03", "16,50", numero(pp["Amount"]["cuantiles"]["0.5"])],
["Máximo", "29.484,80", "3.188,80", "38.536,74"],
["Importe cero: filas", "1.071", "12", entero(pp["Amount"]["ceros"])],
["Suma observada", numero(clases["0"]["amount_total"]), numero(clases["1"]["amount_total"]), numero(res["conjuntos"]["production"]["amount_total"])]]) + """
<p>La suma de Amount fraudulento es 52.878,99 en unidades monetarias no especificadas. Es volumen asociado a etiquetas de fraude,
no pérdida económica demostrada, ahorro recuperable ni impacto de un detector.</p>
""" + tabla(["Tramo Amount (límite superior incluido)", "Transacciones", "Fraudes", "% fraude"],
[[t["tramo"], entero(t["filas"]), entero(t["fraudes"]), porcentaje(t["tasa_fraude"])] for t in importe["tramos"]]) +
tabla(["Cola superior", "Umbral", "Filas por encima", "Fraudes", "% fraude"],
[[f"p{numero(t['cuantil'] * 100, 1)}", numero(t["limite"]), entero(t["filas"]), entero(t["fraudes"]), porcentaje(t["tasa_fraude"])] for t in importe["colas_amount"]]) + f"""
<p>Una regla que marque cualquier extremo de Tukey en al menos una variable seleccionaría <strong>67.908 filas ({porcentaje(67908/170884, 2)})</strong>.
Capturaría 284 de los 295 fraudes (recall 96,27 %), pero su precisión sería solo 0,418 %: produciría 67.624 falsas alertas en construcción.
Es un diagnóstico sobre estos mismos datos, no una regla validada. Eliminar automáticamente esos extremos borraría casi toda la clase positiva.</p>
""")

pares_x = [p for p in corr["construccion"]["pares_principales"] if p["a"] != "Amount" and p["b"] != "Amount"]
seccion("relaciones", "8. Relaciones y dimensionalidad", figura("correlaciones.png", "Pearson sobre todas las filas de construcción, incluidas las fraudulentas.") +
tabla(["Variable A", "Variable B", "Pearson r"], [[p["a"], p["b"], numero(p["r"], 4)] for p in corr["construccion"]["pares_principales"][:10]]) +
f"""<p>Amount concentra las correlaciones lineales globales más importantes: X2 (−0,532), X20 (0,372), X7 (0,366) y X5 (−0,362).
X1–X20 presentan correlaciones globales pequeñas. Este patrón es compatible con variables previamente transformadas,
pero no demuestra que sean componentes PCA ni permite reconstruir su significado.</p>
<p>La matriz de correlación de los 21 predictores necesita {corr['construccion']['componentes_95_pct_varianza_estandarizada']} componentes para explicar el 95 % de la varianza estandarizada.
Reducir dimensionalidad no aparece como una necesidad inmediata solo por este criterio. Esto es un diagnóstico de varianza, no de preservación de señal de fraude.</p>
<p>En la unión de construcción y producción, X1–X20 están casi perfectamente centradas y decorrelacionadas:
la mayor correlación absoluta es 0,00002584 y sus desviaciones decrecen estrictamente desde X1 hasta X20.
La matriz estandarizada tiene rango 20; las 21 variables, incluyendo Amount, tienen rango 21. No hay columnas idénticas.
Esta estructura sugiere un procesamiento previo, cuya técnica y población de ajuste no están documentadas. No se puede confirmar ni descartar
uso previo de producción al construir esa transformación.</p>
<p>Un ajuste lineal puramente descriptivo de Amount sobre X1–X20 en la unión explica el 88,607 % de su varianza.
Queda un 11,393 % residual: Amount comparte información, pero no es una combinación lineal exacta. Este ajuste sobre los mismos datos no es una prueba de generalización.</p>
<p>La correlación cambia al condicionar por clase. Se han guardado matrices Pearson y Spearman separadas para construcción, producción, normales y fraudes.
La baja correlación global no equivale a independencia ni garantiza baja dependencia dentro de la minoría.</p>""" +
tabla(["A", "B", "Pearson dentro de los 295 fraudes"], [[p["a"],p["b"],numero(p["r"],4)] for p in corr["fraude"]["pares_principales"][:8]]))

dupf = []
for nombre, bloque in [("Construcción",exactos["construccion"]),("Producción",exactos["produccion"])]:
    dupf.append([nombre, entero(bloque["patrones_unicos"]), entero(bloque["grupos_duplicados"]),
                entero(bloque["filas_en_grupos_duplicados"]), entero(bloque["filas_redundantes_tras_primera"]), entero(bloque["maxima_multiplicidad"])])
split = exactos["split_aleatorio_notebook"]
alt = exactos["alternativa_agrupada"]
cruce = exactos["coincidencias_entre_datasets"]
seccion("duplicados", "9. Duplicados, coincidencias y validación", """
<p>Se ha comparado la igualdad numérica de las columnas originales mediante agrupaciones y cruces, no únicamente hashes.
Un <em>patrón</em> es una combinación de los 21 predictores observados. Las filas adicionales son las que quedan al conservar solo la primera de cada patrón;
las filas en grupos repetidos incluyen también la primera.</p>
""" + tabla(["Conjunto", "Patrones únicos", "Grupos repetidos", "Filas en esos grupos", "Filas adicionales", "Máxima repetición"], dupf) + """
<p>Los 295 fraudes de construcción contienen <strong>285 patrones únicos</strong>. Hay 16 filas fraudulentas pertenecientes a 6 grupos repetidos, con 10 repeticiones adicionales.
No se observan patrones de construcción con etiquetas contradictorias, ni usando las 21 variables ni usando solo X1–X20.</p>
<p><strong>Igualdad de predictores no prueba identidad de transacción.</strong> La cuantización puede hacer coincidir observaciones distintas.
Por ello, el estudio no borra filas ni copia etiquetas entre archivos. Producción debe conservar sus 113.923 filas, incluso las repetidas.</p>
""" + tabla(["Cruce construcción–producción", "Resultado"], [
["Patrones compartidos de 21 variables", entero(cruce["patrones_presentes_en_ambos"])],
["Filas de producción que coinciden con construcción", entero(cruce["filas_produccion_con_coincidencia"]) + " (" + numero(cruce["porcentaje_produccion_con_coincidencia"],3) + " %)"],
["Filas de construcción implicadas", entero(cruce["filas_construccion_con_coincidencia"])],
["Filas de producción coincidentes con un patrón fraudulento de construcción", entero(cruce["filas_produccion_coincidentes_con_patron_fraudulento_construccion"]) + "; su etiqueta real sigue siendo desconocida"],
["Filas de producción coincidentes si se omite Amount", entero(reducidos["coincidencias_entre_datasets"]["filas_produccion_con_coincidencia"])]]) +
"<h3>Reproducción exacta del split del notebook</h3>" + tabla(["Partición", "Filas train", "Fraudes train", "Filas validación", "Fraudes validación", "Filas validación vistas en train"], [
["Aleatoria estratificada 70/30, semilla 42", entero(split["filas_entrenamiento"]), entero(split["fraudes_entrenamiento"]), entero(split["filas_validacion"]), entero(split["fraudes_validacion"]), entero(split["filas_validacion_con_predictores_en_entrenamiento"])],
["Alternativa agrupada por 21 predictores", entero(alt["filas_entrenamiento"]), entero(alt["fraudes_entrenamiento"]), entero(alt["filas_validacion"]), entero(alt["fraudes_validacion"]), "0"]]) + f"""
<p>El split del notebook reparte <strong>1.213 patrones</strong> entre ambos lados: <strong>1.643 filas de validación (3,205 %)</strong> tienen los mismos predictores que alguna fila de entrenamiento.
Incluye <strong>5 de los 89 fraudes de validación</strong> (5,618 %). Sin Amount, el solape sube a 2.232 filas.</p>
<p>Esto crea un riesgo concreto de optimismo al medir generalización a patrones nuevos. No se ha medido cuánto cambia AP por ese solape.
En el GMM inicial, los fraudes de entrenamiento no se usan para ajustar el modelo: esos cinco positivos coincidentes no implican que el GMM los haya aprendido como fraudes;
las 1.638 coincidencias normales sí pueden estar en su muestra de ajuste. El riesgo y su efecto dependen del método.</p>
<p>Se ha comprobado una alternativa reproducible con StratifiedGroupKFold, 10 folds, mezcla y semilla 42, asignando los folds 0, 1 y 2 a validación.
Da {entero(alt['filas_validacion'])} filas y {entero(alt['fraudes_validacion'])} fraudes en validación, sin grupos solapados.
Es una sensibilidad por patrón observable; no sustituye grupos de cliente o fechas, que no están disponibles.</p>
<p>Los fraudes por decil de posición física son 25, 26, 29, 40, 30, 25, 27, 31, 37 y 25.
Las repeticiones aparecen a distancia en el archivo, no solo en filas adyacentes. No hay fundamento documental para tratar el orden como cronología.</p>
<p><a href="https://scikit-learn.org/stable/modules/cross_validation.html">La documentación de validación cruzada</a> explica la separación de grupos y la estratificación.
La agrupación permite evaluar patrones no vistos, mientras una partición aleatoria puede ser más próxima a un escenario donde los patrones se repiten.
Conviene declarar cuál es el objetivo y presentar la sensibilidad, sin escoger el split por la puntuación más favorable.</p>
""")

seccion("deriva", "10. Construcción frente a producción", """
<p>Las distribuciones marginales observadas son muy parecidas. El mayor KS es 0,00501 en X8: una diferencia máxima de aproximadamente 0,50 puntos porcentuales entre distribuciones acumuladas.
El mayor PSI por deciles es 0,000246 en X1. No hay evidencia fuerte de deriva marginal con estos diagnósticos.</p>
""" + tabla(["Variable", "KS", "p KS ajustado BH", "PSI", "Δ media / SD const.", "Prod. fuera rango const."],
[[d["variable"],numero(d["ks"],6),numero(d["p_ks_bh"],4),numero(d["psi_decil_suavizado"],6),numero(d["diferencia_medias_en_sd_construccion"],5),entero(d["produccion_fuera_rango_const"])] for d in drift]) + """
<p>KS se calcula con todas las filas y aproximación asintótica. Los valores p se ajustan por Benjamini–Hochberg para 21 comparaciones.
Los empates por cuantización y la posible dependencia entre observaciones con patrones repetidos limitan la interpretación inferencial de estos valores p; el foco son las magnitudes de las diferencias.
PSI usa deciles de construcción, extremos abiertos y suavizado de 0,5 por celda. No se aplica un umbral universal de PSI.</p>
<p>Amount máximo aumenta de 29.484,80 a 38.536,74 y su desviación pasa de 363,49 a 392,06. Hay valores nuevos fuera del rango de construcción en algunas variables,
aunque la distribución central sea parecida. Un preprocesamiento futuro debe poder transformar esos extremos sin fallar ni recortarlos de forma improvisada.</p>
<h3>Diagnóstico multivariante de origen</h3>
<div id="resultado-multivariante"></div>
<p>Esta prueba intenta distinguir el archivo de procedencia a partir de los predictores, con evaluación separada por grupos.
No predice fraude. Un resultado cercano al azar solo limita la evidencia de shift detectable por este modelo y este split;
no demuestra identidad de distribuciones, ausencia de cambios en P(Class|X) ni la misma prevalencia en producción.</p>
<details><summary>Distribuciones completas por conjunto</summary>""" + figura("distribuciones_conjuntos.png", "Comparación de las 21 variables; mismo rango y normalización por conjunto. Las colas fuera de la vista se conservan en las estadísticas.") + "</details>")

seccion("notebook", "11. Auditoría completa del notebook", """
<p>Revisión estática de las 25 celdas y sus salidas guardadas. Las salidas históricas están fechadas el 15/10/2026,
entre las 17:33 y las 18:16 UTC. No se ha ejecutado ni modificado el notebook original.</p>
<div class="hallazgo"><strong>El pipeline está incompleto.</strong> Se carga y explora, se crea un split y se ajusta un GMM con 5 componentes sobre normales de train.
No se calcula AP, no se compara con referencias, no se infiere producción y no se exporta.</div>
<p>En las celdas 22–24, <code>score_samples</code> devuelve log-densidad bajo el modelo ajustado a normalidad.
Para usar ese candidato como score de anomalía, el sentido coherente es <code>-gm_model.score_samples(X)</code>.
No es una probabilidad de fraude calibrada. Ordenar para inspeccionar extremos es válido, pero una exportación debe mantener el orden original de producción.
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html">Referencia de GaussianMixture</a>.</p>
""" + tabla(["Celda (base 1)", "Tipo", "Contador", "Contenido y revisión"],
[[c["celda"],c["tipo"],c["execution_count"] if c["execution_count"] is not None else "—",c["revision"]] for c in nb["celdas"]]) + """
<h3>Reproducibilidad y entorno</h3>
<p>El notebook depende de pandas, NumPy, tqdm, Matplotlib, scikit-learn, IPython y Plotly; además, de Jupyter/IPython para las instrucciones <code>!wget</code> y <code>!unzip</code>.
No hay versiones fijadas. Parte de los imports está sin uso: las métricas se importan pero no se calculan.</p>
<p><code>COLAB=True</code> no detecta el entorno y fuerza el camino de descarga. Una ejecución local puede sobreescribir ZIP o solicitar confirmación al extraer archivos existentes.
La futura versión debería leer las copias locales ya verificadas. El scatter usa una muestra de 10.000 filas sin semilla:
el gráfico guardado contiene solo 16 fraudes y estudia únicamente X1/X2.</p>
<p>Los contadores no corresponden a una ejecución lineal: por ejemplo, la celda 20 tiene contador 31 y la 21 contador 28.
No hay tracebacks guardados. Los conteos y desviaciones visibles son coherentes, pero el historial no prueba reproducibilidad desde un kernel limpio.
El metadato Python 3.6.5 y el enlace a sklearn 1.6 en la representación del estimador no constituyen un manifiesto fiable del entorno.</p>
<p>El resumen automático auxiliar de Colab en la celda 24 contiene una desviación de la etiqueta igual a cero pese a mostrar ambas clases.
Se descarta ese resumen para el estudio; las estadísticas se recalculan desde los datos reales.</p>
""")

seccion("metodo", "12. Propuesta de trabajo basada en los hallazgos", """
<ol>
<li><strong>Fijar protocolo antes de comparar modelos.</strong> Reservar una evaluación final y usar validación interna agrupada para selección.
El estudio ya ha observado toda construcción: una partición creada ahora no sería un conjunto nunca visto por el analista.
Producción conserva etiquetas ocultas y no puede usarse para elegir hiperparámetros por AP.</li>
<li><strong>Medir AP como criterio principal.</strong> Usar scores orientados a fraude, comprobar prevalencia como referencia y añadir curva PR.
ROC-AUC, recall o precision@k pueden completar la lectura, pero no sustituir la métrica solicitada.
Con 89 positivos en el split original, un fraude adicional cambia el recall en 1,12 puntos porcentuales: pequeñas diferencias merecen cautela.</li>
<li><strong>Establecer referencias sencillas.</strong> Score constante y variables univariantes seleccionadas dentro de train; después reproducir el GMM de partida con signo correcto,
convergencia revisada y preprocesamiento ajustado solo en train. Los números descriptivos de este informe no se reutilizan como resultados validados.</li>
<li><strong>Comparar familias justificadas.</strong> Un método de normalidad como GMM o Isolation Forest y un clasificador supervisado regularizado son candidatos razonables,
ya que hay etiquetas y el enunciado no prohíbe supervisión. Comparar con el mismo protocolo y recursos acotados; no hay evidencia aquí para proclamar un ganador.</li>
<li><strong>Evaluar transformaciones, no asumirlas.</strong> Probar escala original, escalado y log1p de Amount cuando el método lo justifique.
Todo transformador, remuestreo o selección debe ajustarse dentro de cada fold. No descartar extremos ni crear un dataset balanceado de validación:
alteraría la población y la lectura de precisión/AP.</li>
<li><strong>Mostrar sensibilidad y estabilidad.</strong> Comparar resultados por folds y por solape de patrones, y utilizar remuestreo por grupos para incertidumbre si se estima.
Mantener la distinción entre nuevos patrones y transacciones que repiten patrones conocidos.</li>
<li><strong>Preparar la entrega tras elegir configuración.</strong> Ajustar el modelo final en construcción, separar Class de producción,
generar exactamente 113.923 scores finitos, una sola columna y el orden original. Exportar sin índice y resolver la convención de cabecera del curso.</li>
<li><strong>Completar la explicación académica.</strong> Incluir integrantes, objetivo, datos, protocolo, decisiones, resultados medidos, limitaciones y reflexión final.
Ejecutar entonces el notebook final de arriba abajo y conservar sus salidas visibles.</li>
</ol>
<p>Las <a href="https://scikit-learn.org/stable/common_pitfalls.html#data-leakage">recomendaciones oficiales sobre fuga de datos</a>
respaldan separar los datos antes de aprender transformaciones. Esta propuesta no modifica los datos ni sustituye la entrega del taller.</p>
""")

seccion("limites", "13. Qué queda sin resolver", """
<ul>
<li>No se conoce la etiqueta de ninguna fila de producción. No se puede calcular su AP, su tasa real de fraude ni un coste de error.</li>
<li>No se conocen el periodo, moneda, país, entidad, cliente o comercio. No se puede estudiar temporalidad, reincidencia, cambios por cliente ni validación cronológica.</li>
<li>No está documentado el significado ni la transformación de X1–X20. La interpretación es estadística, no causal ni de negocio.</li>
<li>Las coincidencias de predictores no permiten afirmar que se repita la misma operación real. La ausencia de etiquetas contradictorias tampoco garantiza etiquetas perfectas.</li>
<li>La separación univariante se ha medido dentro de construcción. No acredita el rendimiento generalizable de un modelo. Las repeticiones podrían reducir la independencia efectiva si corresponden a observaciones relacionadas.</li>
<li>Las pruebas de distribución no certifican estabilidad de la relación entre variables y fraude. Cambios raros o subpoblaciones pequeñas pueden escapar al diagnóstico.</li>
<li>El GMM guardado no se ha reentrenado, y no existe una comparativa validada ni un score final de producción.</li>
<li>La cabecera del CSV y la precisión completa de la fecha de entrega deben contrastarse con las indicaciones del aula al cerrar la entrega.</li>
</ul>
""")

seccion("reproducir", "14. Evidencias y reproducción", """
<p>El HTML incorpora las imágenes y funciona sin conexión. El directorio del estudio contiene los scripts de análisis y los resultados JSON,
con perfiles de todos los campos, matrices, multiplicidades, diagnósticos y versiones. Los archivos originales quedan intactos.</p>
""" + tabla(["Archivo de estudio", "Contenido"], [
["perfilar_datos.py", "Perfiles completos, clases, extremos, correlaciones, comparación marginal y figuras"],
["auditar_duplicados.py", "Duplicados exactos, coincidencias, cuantización y particiones"],
["auditar_deriva_multivariante.py", "Modelo de origen y diagnóstico estructural"],
["generar_informe.py", "Compone este informe desde las evidencias"],
["requirements-estudio.txt", "Versiones del entorno usado para los análisis nuevos"],
["auditoria_notebook.json", "Revisión estática por celda y hallazgos"],
["manifiesto_fuentes.json", "Inventario de entrada y SHA-256"],
["resumen_datos.json y perfiles_columnas.json", "Perfiles de todas las columnas y resumen"],
["perfiles_por_clase.json y senales_univariantes.json", "Perfiles y separación marginal de clases"],
["correlaciones.json", "Matrices Pearson/Spearman y autovalores"],
["importes_extremos.json y orden_filas.json", "Importes, colas, reglas descriptivas y bloques de orden"],
["comparacion_distribuciones.json y deriva_multivariante.json", "Diferencias entre conjuntos"],
["duplicados_validacion.json", "Patrones repetidos y solape del split"]]) + """
<p>Desde la raíz del proyecto y con un entorno que tenga las dependencias indicadas:</p>
<pre>python estudio_datos/perfilar_datos.py
python estudio_datos/auditar_duplicados.py
python estudio_datos/auditar_deriva_multivariante.py
python estudio_datos/generar_informe.py</pre>
<p>Los tres análisis numéricos leen los ZIP; el generador compone resultados y la auditoría estática conservada. La auditoría documental y del notebook
requiere revisión nueva si cambian esos originales. Los scripts no modifican el notebook ni los ZIP.</p>
""" + tabla(["Componente", "Versión medida en este estudio"], [["Python",res["python"]]] + list(res["versiones"].items())) + """
<p>Estas versiones corresponden al entorno de análisis del 19/10/2026, no certifican el entorno con que se generaron las salidas históricas del notebook.</p>
""")

css = """
:root{--ink:#193047;--muted:#516578;--line:#d8e1e8;--blue:#265d8f;--pale:#eff5fa}
*{box-sizing:border-box}body{margin:0;background:#f7f9fb;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1220px;margin:36px auto;background:white;padding:48px 58px;border:1px solid var(--line)}
header{border-bottom:3px solid var(--blue);padding-bottom:24px}header p{color:var(--muted);max-width:870px}
h1{font-size:38px;line-height:1.2;margin:8px 0 20px;letter-spacing:-.6px}h2{font-size:27px;line-height:1.25;margin:0 0 22px}h3{font-size:20px;margin:26px 0 12px}
section{padding:36px 0;border-bottom:1px solid var(--line);scroll-margin-top:25px}p{margin:14px 0}a{color:#205c90;text-underline-offset:3px}
nav{padding:24px;background:var(--pale);margin:28px 0}nav ol{columns:2;column-gap:45px;margin:0;padding-left:22px}nav li{break-inside:avoid;padding:3px 0}
.tabla{overflow-x:auto;margin:20px 0;border:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums}
th{text-align:left;background:#eaf1f7;color:#17334e;font-weight:650}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:nth-child(even){background:#f8fafc}td:first-child{font-weight:550}td{overflow-wrap:anywhere}tr:last-child td{border-bottom:0}
figure{margin:28px 0}figure img{display:block;max-width:100%;height:auto}figcaption{font-size:13px;color:var(--muted);padding-top:8px}
.hallazgo{border-left:4px solid var(--blue);padding:17px 22px;background:var(--pale);margin:22px 0}li{padding:6px 0}details{border:1px solid var(--line);padding:14px 18px;margin:20px 0}
summary{cursor:pointer;font-weight:600}code{background:#edf2f7;padding:2px 5px;border-radius:3px;font-size:.9em}pre{overflow:auto;padding:20px;background:#edf2f7;font-size:13px;line-height:1.6}
.eyebrow{color:#45637d;font-size:14px;letter-spacing:.08em;text-transform:uppercase}footer{color:var(--muted);font-size:13px;padding-top:24px}
@media(max-width:760px){main{margin:0;padding:28px 20px;border:0}h1{font-size:29px}h2{font-size:23px}nav ol{columns:1}th,td{padding:8px}table{min-width:620px}}
@media print{body{background:white}main{max-width:none;margin:0;padding:0;border:0}nav{display:none}section{break-before:auto}h2,h3{break-after:avoid}.tabla{overflow:visible}table{font-size:9px}figure{break-inside:avoid}details{display:block}summary{font-size:14px}}
"""

# Resumen legible del diagnóstico de origen, separado de cualquier métrica de fraude.
ev = multi["evaluacion"]
part = multi["particion"]
multi_html = tabla(["Diagnóstico", "Resultado"], [
    ["Modelo", "HistGradientBoostingClassifier; 100 iteraciones, 15 hojas; sin ajuste sobre evaluación"],
    ["Población", "284.807 filas; los 21 predictores; excluye Class e índice"],
    ["Partición", "GroupShuffleSplit por patrón exacto; semilla 20261019"],
    ["Train / evaluación", entero(part["entrenamiento"]["filas"]) + " / " + entero(part["evaluacion"]["filas"])],
    ["Grupos compartidos", entero(part["grupos_compartidos_train_test"])],
    ["ROC AUC de origen", numero(ev["roc_auc_origen"], 6)],
    ["IC95 % bootstrap de grupos", "[" + numero(ev["bootstrap"]["ic95"][0],6) + "; " + numero(ev["bootstrap"]["ic95"][1],6) + "]"],
    ["Log loss / referencia constante", numero(ev["log_loss"],6) + " / " + numero(ev["log_loss_referencia_constante"],6)],
    ["Brier / referencia constante", numero(ev["brier"],6) + " / " + numero(ev["brier_referencia_constante"],6)],
]) + '<p>El intervalo usa 500 remuestreos de grupos manteniendo fijo el modelo. No incorpora variación entre particiones ni nuevos entrenamientos. El modelo no supera la referencia constante en log loss o Brier.</p>'
fragmentos = "".join(s[2] for s in secciones).replace('<div id="resultado-multivariante"></div>', multi_html)
indice = "".join(f'<li><a href="#{s[0]}">{s[1].split(". ",1)[1]}</a></li>' for s in secciones)
html = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Estudio de datos · Taller de detección de anomalías</title><style>{css}</style></head><body><main>
<header><div class="eyebrow">Taller de detección de anomalías · 19 de octubre de 2026</div>
<h1>Estudio de archivos, requisitos y datos</h1><p>Fraude en tarjetas de crédito. Revisión del enunciado y del notebook, análisis íntegro de los datos y propuesta de evaluación basada en los hallazgos.</p></header>
<nav aria-label="Índice"><ol>{indice}</ol></nav>{fragmentos}<footer>Fuentes: PDF y notebook locales, ambos CSV comprimidos y documentación oficial enlazada. Análisis descriptivo y de integridad; sin entrega de predicciones ni modificación de los originales.</footer>
</main></body></html>'''
(BASE / "informe_estudio.html").write_text(html, encoding="utf-8")
print(f"Informe creado: {BASE / 'informe_estudio.html'} ({len(html.encode()):,} bytes)")
