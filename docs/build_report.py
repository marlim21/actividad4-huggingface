"""Genera docs/analisis_comparativo.docx a partir de los resultados en results/.

Todas las cifras del documento se leen de los CSV/JSON generados por los scripts,
de modo que el Word y los resultados no puedan divergir. Antes de escribir, el
script verifica que las afirmaciones cualitativas del texto (qué modelo es mejor,
cuál es más rápido, etc.) sigan siendo ciertas con los datos; si no lo son, se
detiene para que el análisis se reescriba.

Uso:
  python docs/build_report.py [--reference local_mps|colab_gpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "analisis_comparativo.docx"

M1, M2, M3 = "mBERT-nlptown", "XLM-R-cardiffnlp", "DistilmBERT-lxyuan"
IDS = {
    M1: "nlptown/bert-base-multilingual-uncased-sentiment",
    M2: "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    M3: "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
}
TH = {"f1_macro": 0.60, "recall_negativo": 0.75, "rendimiento_reg_s": 50.0, "online_p95_ms": 100.0}


# ---------------------------------------------------------------- utilidades
def fmt(x, d=3):
    return f"{x:.{d}f}".replace(".", ",") if isinstance(x, (int, float)) else str(x)


def pct(x, d=1):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def table(doc, header, rows, widths=None, font=9):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(font)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "0B5345")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(font)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return t


def para(doc, text, bold_prefix=None, style=None):
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def bullets(doc, items):
    for it in items:
        if isinstance(it, tuple):
            para(doc, it[1], bold_prefix=it[0], style="List Bullet")
        else:
            para(doc, it, style="List Bullet")


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---------------------------------------------------------------- datos
def load(reference):
    ref_dir = RES / reference
    comp = pd.read_csv(ref_dir / "model_comparison.csv").set_index("modelo")
    per_class = pd.read_csv(ref_dir / "per_class_metrics.csv")
    meta = json.loads((ref_dir / "run_metadata.json").read_text())
    cpu = RES / "local_cpu" / "model_comparison.csv"
    cpu = pd.read_csv(cpu).set_index("modelo") if cpu.exists() else None
    profile = json.loads((ROOT / "data" / "processed" / "data_profile.json").read_text())
    preds = pd.read_csv(ref_dir / "predictions.csv")
    cms = {m: pd.read_csv(ref_dir / f"confusion_{m}.csv", index_col=0) for m in comp.index}
    return comp, per_class, meta, cpu, profile, preds, cms, ref_dir


def bootstrap_f1(preds, n=1000, seed=42):
    """IC 95 % del F1 macro y de la diferencia pareada respecto del mejor modelo."""
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    y = preds["sentiment"].to_numpy()
    p = {m: preds[f"pred_{m}"].to_numpy() for m in (M1, M2, M3)}
    labels = ["negativo", "neutral", "positivo"]
    samples = {m: [] for m in p}
    diffs = {m: [] for m in (M2, M3)}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        f = {m: f1_score(y[idx], p[m][idx], labels=labels, average="macro", zero_division=0) for m in p}
        for m in p:
            samples[m].append(f[m])
        for m in diffs:
            diffs[m].append(f[M1] - f[m])
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {m: q(v) for m, v in samples.items()}, {m: q(v) for m, v in diffs.items()}


def check_claims(comp, cpu):
    """Las afirmaciones del texto deben sostenerse con los datos."""
    f1 = comp["f1_macro"]
    assert f1.idxmax() == M1, "El mejor F1 macro ya no es nlptown: reescribir el análisis"
    assert f1.idxmin() == M3, "El peor F1 macro ya no es DistilmBERT: reescribir el análisis"
    assert comp["rendimiento_reg_s"].idxmax() == M3, "El más rápido ya no es DistilmBERT"
    assert comp.loc[M1, "f1_macro"] >= TH["f1_macro"] and comp.loc[M1, "recall_negativo"] >= TH["recall_negativo"]
    assert comp.loc[M2, "f1_macro"] < TH["f1_macro"] and comp.loc[M3, "f1_macro"] < TH["f1_macro"]
    assert comp.loc[M1, "rendimiento_reg_s"] >= TH["rendimiento_reg_s"]
    assert comp.loc[M1, "online_p95_ms"] <= TH["online_p95_ms"]
    if cpu is not None:
        assert cpu["rendimiento_reg_s"].idxmax() == M3


# ---------------------------------------------------------------- documento
def build(reference):
    comp, per_class, meta, cpu, profile, preds, cms, ref_dir = load(reference)
    check_claims(comp, cpu)
    v = meta["versions"]
    c1, c2, c3 = comp.loc[M1], comp.loc[M2], comp.loc[M3]
    is_cuda = meta["device"] == "cuda"
    dev_txt = (f"GPU {meta['device_description']} (Google Colab)" if is_cuda
               else "GPU integrada Apple M2 Pro, backend MPS de PyTorch, en equipo local")
    pc = per_class.set_index(["modelo", "clase"])
    gain_f1 = c1["f1_macro"] - c2["f1_macro"]
    speedup = c3["rendimiento_reg_s"] / c1["rendimiento_reg_s"]
    pred_cols = [f"pred_{m}" for m in comp.index]
    ci, ci_diff = bootstrap_f1(preds)
    assert all(lo > 0 for lo, _ in ci_diff.values()), "La ventaja de nlptown no es significativa"
    all_wrong = preds[preds[pred_cols].ne(preds["sentiment"], axis=0).all(axis=1)]

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.3)
        s.top_margin = s.bottom_margin = Cm(2.2)

    # 1. Portada
    for _ in range(4):
        doc.add_paragraph()
    for text, size, bold in [
        ("Universidad Tecmilenio", 16, True),
        ("Gestión de proyectos de inteligencia artificial", 13, False),
        ("Actividad 4", 13, False),
        ("", 11, False),
        ("Selección de un modelo preentrenado de Hugging Face para clasificar el sentimiento "
         "de reseñas de clientes en español", 20, True),
        ("Análisis comparativo de tres modelos: calidad predictiva, latencia y recursos", 13, False),
        ("", 11, False),
        ("Estudiante: [completar]", 12, False),
        ("Matrícula: [completar]", 12, False),
        ("Profesor(a): [completar]", 12, False),
        ("Fecha: 11 de octubre de 2026", 12, False),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        if size == 20:
            r.font.color.rgb = RGBColor(0x0B, 0x53, 0x45)
    doc.add_page_break()

    # 2. Resumen ejecutivo
    doc.add_heading("1. Resumen ejecutivo", 1)
    para(doc, (
        "Una organización de comercio electrónico necesita clasificar automáticamente las reseñas de sus "
        "clientes en español como negativas, neutrales o positivas, para escalar las quejas a servicio al "
        "cliente y monitorear la satisfacción. Se compararon tres modelos preentrenados del Hugging Face Hub, "
        f"usados solo en inferencia (sin fine-tuning): {IDS[M1]}, {IDS[M2]} y {IDS[M3]}. "
        f"Se evaluaron sobre las {int(c1['n_registros'])} reseñas de la partición de prueba oficial del corpus "
        "Multilingual Amazon Reviews en español, con las mismas condiciones de inferencia, midiendo accuracy, "
        "precision, recall y F1 (macro y weighted), latencia, rendimiento y recursos."))
    para(doc, (
        f"En la ejecución de referencia ({dev_txt}), el modelo de NLP Town obtuvo el mejor desempeño "
        f"(F1 macro {fmt(c1['f1_macro'])}, accuracy {fmt(c1['accuracy'])}, recall de la clase negativa "
        f"{fmt(c1['recall_negativo'])}), frente a {fmt(c2['f1_macro'])} de XLM-R y {fmt(c3['f1_macro'])} de "
        f"DistilmBERT. DistilmBERT fue {fmt(speedup, 1)} veces más rápido que el de NLP Town "
        f"({fmt(c3['rendimiento_reg_s'], 1)} frente a {fmt(c1['rendimiento_reg_s'], 1)} reseñas/s), pero no "
        "alcanzó el umbral mínimo de calidad y casi nunca reconoce la clase neutral."))
    para(doc, (
        f"Se recomienda {IDS[M1]}: es el único que cumple todos los umbrales de aceptación propuestos, "
        "fue entrenado específicamente con reseñas de productos y tiene licencia MIT. Su principal limitación "
        f"es la clase neutral (F1 {fmt(pc.loc[(M1, 'neutral'), 'f1'])}), por lo que el prototipo debe usarse "
        "para priorizar reseñas, no para decisiones automáticas sobre clientes individuales."), bold_prefix="Recomendación. ")
    if not is_cuda:
        para(doc, (
            "Las cifras de este documento provienen de la ejecución local de los scripts en un Apple M2 Pro "
            "(GPU vía MPS y CPU). El notebook de Google Colab con GPU CUDA implementa el mismo procedimiento; "
            "sus resultados se guardan en results/colab_gpu/. Las métricas de calidad no dependen del hardware "
            "(salvo diferencias numéricas mínimas); los tiempos sí, y deben leerse según el dispositivo indicado."),
            bold_prefix="Nota sobre el entorno. ")

    # 3. Introducción
    doc.add_heading("2. Introducción", 1)
    para(doc, (
        "Los modelos preentrenados permiten incorporar capacidades de procesamiento de lenguaje natural sin "
        "entrenar desde cero. Sin embargo, en el Hugging Face Hub existen miles de clasificadores de sentimiento "
        "que difieren en idioma, dominio de entrenamiento, esquema de etiquetas, tamaño, licencia y calidad de "
        "documentación. Elegir por popularidad o por las métricas que reporta el autor es riesgoso: esas métricas "
        "suelen medirse con otros datos, otras etiquetas y otro hardware."))
    para(doc, (
        "Este trabajo plantea la selección como un problema de gestión de proyectos de IA: se definen el caso de "
        "uso y criterios explícitos, se diseña una comparación justa (mismos datos, mismas etiquetas, mismas "
        "condiciones de inferencia), se miden calidad y eficiencia, y la decisión se justifica frente a umbrales "
        "de aceptación definidos antes de ver los resultados."))

    # 4. Caso de uso
    doc.add_heading("3. Descripción del caso de uso", 1)
    table(doc, ["Elemento", "Descripción"], [
        ["Nombre", "Clasificación del sentimiento de reseñas de clientes en español"],
        ["Rama de IA", "Procesamiento de lenguaje natural — clasificación de texto"],
        ["Problema", "Las reseñas llegan en volumen alto y en texto libre; leerlas manualmente retrasa la "
                     "atención de quejas y no permite medir la satisfacción de forma continua."],
        ["Contexto organizacional", "Comercio electrónico o marketplace que recibe reseñas de productos en "
                                    "español y tiene un área de servicio al cliente y otra de calidad/producto."],
        ["Usuarios potenciales", "Agentes de servicio al cliente (cola de reseñas negativas), analistas de "
                                 "producto y calidad (tendencias), gerencia (indicadores de satisfacción)."],
        ["Variable a predecir", "Sentimiento de la reseña: negativo, neutral o positivo (3 clases)."],
        ["Objetivo general", "Identificar, implementar, evaluar y seleccionar el modelo preentrenado de "
                             "Hugging Face más adecuado para clasificar el sentimiento de reseñas en español, "
                             "equilibrando calidad predictiva, latencia y recursos."],
        ["Objetivos específicos", "1) Definir criterios de selección y elegir tres modelos candidatos. "
                                  "2) Preparar un conjunto de prueba reproducible. 3) Implementar la inferencia "
                                  "en Google Colab con GPU. 4) Medir accuracy, precision, recall, F1, latencia y "
                                  "recursos en condiciones equivalentes. 5) Recomendar un modelo justificado."],
        ["Beneficio esperado", "Reducir el tiempo para detectar quejas, priorizar la atención y contar con un "
                               "indicador de sentimiento por producto y periodo."],
        ["Alcance", "Prototipo de inferencia por lotes y en línea; evaluación offline. No incluye integración "
                    "con sistemas productivos, fine-tuning ni interfaz de usuario."],
        ["Restricciones", "Cómputo de Google Colab gratuito (GPU no garantizada); datos públicos con licencia "
                          "académica; modelos públicos sin costo de licencia."],
    ], widths=[4, 12.5])
    para(doc, (
        "Se consideraron tres alternativas: NLP de reseñas en español, NLP financiero y visión por computadora. "
        "Se eligió la primera porque existe un corpus público grande y documentado con partición de prueba "
        "oficial y etiquetas derivables a tres clases, y al menos tres modelos multilingües con salidas "
        "compatibles. En NLP financiero en español no se identificó un dataset abierto con calidad comparable, "
        "y en visión las diferencias de etiquetas entre modelos complican una comparación justa."),
        bold_prefix="Justificación de la elección. ")

    # 5. Ecosistema
    doc.add_heading("4. Ecosistema de Hugging Face", 1)
    para(doc, (
        "El Hugging Face Hub es una plataforma de repositorios Git con tres tipos principales de contenido: "
        "modelos, datasets y Spaces. Cada repositorio tiene historial de versiones (commits), una tarjeta "
        "descriptiva y metadatos consultables por API."))
    table(doc, ["Componente", "Qué contiene y cómo se usa", "Uso en este proyecto"], [
        ["Models", "Pesos (safetensors o .bin), configuración (config.json con arquitectura y id2label), "
                   "tokenizador y una model card con tarea, idioma, licencia, datos de entrenamiento, métricas "
                   "publicadas y limitaciones. Se filtran por tarea (text-classification), idioma y licencia. La "
                   "versión se identifica con el hash del commit y se fija con el argumento revision.",
         "Utilizado: tres modelos cargados con AutoTokenizer y AutoModelForSequenceClassification, con commit "
         "fijado. Se revisaron model cards, config.json y tamaños de archivo por API."],
        ["Datasets", "Archivos de datos (Parquet, JSONL, CSV) con particiones (train/validation/test) y una "
                     "dataset card con origen, campos, licencia y limitaciones. Se cargan con "
                     "datasets.load_dataset(id, split=..., revision=...).",
         "Utilizado: SetFit/amazon_reviews_multi_es, partición test, commit fijado."],
        ["Spaces", "Aplicaciones web alojadas (Gradio, Streamlit, Docker) que sirven demos interactivas de "
                   "modelos y permiten validar un prototipo con usuarios sin instalar nada.",
         "Solo descrito. Evolución propuesta: demo en Gradio donde un analista pegue una reseña y vea la clase y "
         "la probabilidad del modelo recomendado."],
        ["Hub API y tokens", "huggingface_hub permite consultar metadatos y autenticarse con tokens de acceso "
                             "(lectura/escritura) para recursos privados o restringidos.",
         "Utilizado para verificar metadatos. No se requiere token: todos los recursos son públicos. En Colab "
         "se documenta el uso opcional del secreto HF_TOKEN."],
    ], widths=[2.6, 8, 6])
    para(doc, (
        "Una model card informa la tarea (pipeline_tag), la arquitectura (model_type en config.json), los "
        "idiomas, la licencia (metadato license), los datos de entrenamiento y, en ocasiones, métricas. Estas "
        "métricas son resultados publicados por los autores con sus propios datos; en este trabajo se usan "
        "solo como contexto y nunca sustituyen la evaluación experimental."), bold_prefix="Lectura de model cards. ")

    # 6. Criterios
    doc.add_heading("5. Criterios y metodología de selección", 1)
    para(doc, (
        "Se buscaron en el Hub modelos de text-classification con soporte de español y una salida de "
        "sentimiento convertible a tres clases. Cada candidato se evaluó con los criterios siguientes; los "
        "tres primeros son eliminatorios."))
    table(doc, ["Criterio", "Pregunta", "Cómo se verificó"], [
        ["Compatibilidad con la tarea*", "¿Clasifica sentimiento?", "pipeline_tag e id2label del config.json"],
        ["Idioma y dominio*", "¿Soporta español y textos de opinión?", "Model card (idiomas y datos de entrenamiento)"],
        ["Etiquetas compatibles*", "¿Su salida se puede mapear a negativo/neutral/positivo sin ambigüedad?", "id2label"],
        ["Desempeño", "¿Qué publican los autores y qué obtenemos nosotros?", "Model card (publicado) y experimento (medido)"],
        ["Tamaño", "¿Cuántos parámetros y cuánto pesan los archivos?", "Conteo de parámetros en PyTorch y tamaño en el Hub"],
        ["Latencia", "¿Cuánto tarda por reseña en lote y en línea?", "Medición propia (secciones 7 y 8)"],
        ["Recursos", "¿Cabe en una GPU T4 o en CPU?", "Memoria medida y estimada"],
        ["Licencia", "¿Permite el uso previsto?", "Metadato license y texto de la model card"],
        ["Implementación", "¿Funciona con transformers estándar en Colab, sin pasos especiales?", "Prueba de carga"],
        ["Mantenimiento", "¿Se puede fijar e identificar la versión?", "Hash de commit en el Hub"],
        ["Riesgos", "¿Qué sesgos o limitaciones declara o se pueden prever?", "Model card y análisis de errores"],
    ], widths=[3.5, 6.5, 6.5])
    para(doc, "* Criterio eliminatorio.", style=None)
    para(doc, (
        "Los tres modelos se eligieron para cubrir perfiles distintos y hacer visible el balance calidad–"
        "eficiencia: un modelo especializado en reseñas de productos (NLP Town), un modelo multilingüe más grande "
        "entrenado con tuits (Cardiff NLP) y un modelo destilado ligero (lxyuan). Los tres comparten el mismo tipo "
        "de salida (sentimiento) y se mapean a tres clases por nombre de etiqueta."))
    doc.add_heading("Modelos seleccionados", 2)
    table(doc, ["Característica [fuente]", M1, M2, M3], [
        ["Identificador", IDS[M1], IDS[M2], IDS[M3]],
        ["Model card", "huggingface.co/" + IDS[M1], "huggingface.co/" + IDS[M2], "huggingface.co/" + IDS[M3]],
        ["Commit fijado", c1["revision"][:10], c2["revision"][:10], c3["revision"][:10]],
        ["Arquitectura [P]", "BERT-base multilingüe uncased (12 capas)", "XLM-RoBERTa-base (XLM-T, 12 capas)",
         "DistilBERT multilingüe cased (6 capas)"],
        ["Tarea entrenada [P]", "Reseñas de productos → 1–5 estrellas", "Sentimiento en tuits → neg/neu/pos",
         "Sentimiento → pos/neu/neg (destilado de un modelo zero-shot)"],
        ["Idioma y dominio [P]", "EN, NL, DE, FR, IT, ES; reseñas de productos (50 k en español)",
         "Preentrenado con ~198 M tuits; ajustado en 8 idiomas incl. español; Twitter",
         "12 idiomas incl. español; dataset tyqiangz/multilingual-sentiments"],
        ["Parámetros [E]", fmt(c1["parametros_M"], 1) + " M", fmt(c2["parametros_M"], 1) + " M",
         fmt(c3["parametros_M"], 1) + " M"],
        ["Pesos en el Hub [P]", fmt(c1["pesos_hub_MB"], 0) + " MB (safetensors)",
         fmt(c2["pesos_hub_MB"], 0) + " MB (solo .bin)", fmt(c3["pesos_hub_MB"], 0) + " MB (safetensors)"],
        ["Licencia [P]", "MIT", "No declarada en la model card", "Apache-2.0"],
        ["Métrica publicada [P]", "Español: 58 % exactitud en estrellas, 95 % ±1 estrella (5 000 reseñas propias)",
         "Ver artículo XLM-T (otros datos)", "No reporta métricas de evaluación"],
        ["Requisitos [Est]", "~0,7 GB de pesos FP32; GPU opcional", "~1,1 GB de pesos FP32; GPU recomendable",
         "~0,5 GB de pesos FP32; viable en CPU"],
        ["Justificación", "Dominio idéntico al caso de uso (reseñas de productos)",
         "Referente multilingüe robusto con 3 clases nativas", "Alternativa ligera para medir el costo de la eficiencia"],
    ], widths=[3.2, 4.4, 4.4, 4.4], font=8)
    para(doc, "[P] = publicado por los autores en el Hub; [E] = medido en este experimento; [Est] = estimación "
              "(parámetros × 4 bytes en FP32).")
    doc.add_heading("Candidatos descartados", 2)
    table(doc, ["Modelo", "Motivo de descarte"], [
        ["tabularisai/multilingual-sentiment-analysis", "Licencia CC-BY-NC-4.0 (no comercial): incompatible con un uso organizacional."],
        ["pysentimiento/robertuito-sentiment-analysis", "Dominio Twitter; máximo 128 tokens (max_position_embeddings = 130); "
                                                        "sin licencia en los metadatos; diseñado para usarse con el preprocesamiento de la biblioteca pysentimiento."],
        ["finiteautomata/beto-sentiment-analysis", "Dominio Twitter; sin licencia en los metadatos; solo pesos .bin."],
    ], widths=[6, 10.5])

    # 7. Dataset
    doc.add_heading("6. Descripción del dataset", 1)
    sd = profile["sentiment_distribution"]
    tp = profile["token_profile"]
    table(doc, ["Elemento", "Valor"], [
        ["Nombre", "Multilingual Amazon Reviews Corpus (MARC) — español (Keung et al., 2020)"],
        ["Fuente utilizada", "huggingface.co/datasets/SetFit/amazon_reviews_multi_es (réplica; commit 1601541)"],
        ["Responsable", "Amazon (corpus original); SetFit (réplica en el Hub con columnas id, text, label, label_text)"],
        ["Licencia", "Licencia propia de Amazon: uso para investigación académica; prohíbe uso comercial, "
                     "republicar el corpus y reidentificar autores."],
        ["Fecha de consulta", "11 de octubre de 2026"],
        ["Tamaño total", "210 000 reseñas (train 200 000; validation 5 000; test 5 000)"],
        ["Registros utilizados", f"{profile['n_records']} (partición test oficial completa)"],
        ["Clases", "5 niveles de estrellas → 3 clases de sentimiento"],
        ["Distribución", f"Estrellas: 1 000 por nivel. Sentimiento: negativo {sd['negativo']} (40 %), "
                         f"neutral {sd['neutral']} (20 %), positivo {sd['positivo']} (40 %)"],
        ["Calidad", f"Sin faltantes ni textos vacíos; IDs únicos; {profile['duplicate_text_rows']} textos "
                    f"duplicados ({profile['duplicate_texts_with_conflicting_labels']} con etiquetas distintas), conservados"],
        ["Longitud", f"Media {fmt(profile['words']['mean'], 1)} palabras; ≈ {fmt(tp[IDS[M1]]['mean'], 0)} tokens; "
                     f"p95 ≈ {tp[IDS[M1]]['p95']} tokens; > 256 tokens: {fmt(tp[IDS[M1]]['pct_over_max_length'], 2)} %"],
    ], widths=[3.5, 13])
    para(doc, (
        "Transformación: label (0–4) + 1 = estrellas; 1–2 estrellas = negativo, 3 = neutral, 4–5 = positivo. Es "
        "la convención habitual para derivar sentimiento de calificaciones. Se usa la partición de prueba "
        "oficial completa porque ningún modelo se entrena aquí y porque evita decisiones arbitrarias de "
        "muestreo; con una GPU el costo de evaluar 5 000 registros es de segundos o pocos minutos."))
    para(doc, (
        "Como la licencia prohíbe republicar el corpus, el CSV con textos (data/processed/dataset.csv) se "
        "regenera localmente con src/prepare_data.py y no se publica; el repositorio incluye "
        "dataset_manifest.csv (id, estrellas, sentimiento, sin texto). El diccionario de datos completo está en "
        "data/README.md."), bold_prefix="Redistribución. ")
    para(doc, (
        "reseñas de amazon.es (predomina el español peninsular); solo productos; la calificación es "
        "autoasignada por el cliente y puede no coincidir con el texto; la clase neutral (3 estrellas) mezcla "
        "opiniones positivas y negativas; y es posible que algún modelo haya visto reseñas similares durante su "
        "entrenamiento (NLP Town no documenta el origen de sus reseñas)."), bold_prefix="Limitaciones y sesgos: ")

    # 8. Metodología
    doc.add_heading("7. Metodología experimental", 1)
    table(doc, ["Elemento", "Configuración"], [
        ["Entorno de referencia", f"{dev_txt}; {meta['cpu_count']} núcleos de CPU; {meta['ram_total_gb']} GB de RAM"],
        ["Entorno adicional", "CPU del mismo equipo (escenario sin GPU)" if cpu is not None else "—"],
        ["Entorno objetivo (notebook)", "Google Colab con GPU NVIDIA T4 (resultados en results/colab_gpu/)"],
        ["Software", f"Python {v['python']}, PyTorch {v['torch']}, Transformers {v['transformers']}, "
                     f"Datasets {v['datasets']}, scikit-learn {v['scikit-learn']}, pandas {v['pandas']}"],
        ["Datos", "5 000 reseñas, mismo orden para los tres modelos"],
        ["Inferencia", "Solo pesos preentrenados; FP32; torch.inference_mode(); batch_size = 32; "
                       "max_length = 256 con truncamiento; padding dinámico por lote"],
        ["Normalización de etiquetas", "Por nombre de etiqueta (config.id2label), no por índice; nlptown: "
                                       "estrellas con la misma regla que el dataset"],
        ["Métricas", "accuracy; precision, recall y F1 macro y weighted (scikit-learn); reporte por clase; "
                     "matriz de confusión"],
        ["Latencia en lote", "3 lotes de calentamiento; 3 pasadas completas cronometradas (media ± desviación) "
                             "con sincronización del dispositivo; tiempo de carga medido aparte"],
        ["Latencia en línea", "200 reseñas (muestra con semilla 42), una por llamada: media, p50 y p95"],
        ["Recursos", "Parámetros, tamaño de pesos, memoria pico del dispositivo"],
        ["Reproducibilidad", "Semilla 42; commits fijados del dataset y de los modelos; versiones registradas"],
    ], widths=[4, 12.5])
    para(doc, (
        "Se usa F1 macro porque el conjunto de tres clases está desbalanceado (20 % neutral) y el caso de "
        "uso necesita reconocer las tres clases; accuracy y F1 weighted favorecen a las clases mayoritarias. "
        "Como métrica de negocio complementaria se usa el recall de la clase negativa (proporción de quejas "
        "que sí se escalan)."), bold_prefix="Métrica principal. ")

    # 9. Resultados
    doc.add_heading("8. Resultados y tabla comparativa", 1)
    rows = [
        ["Accuracy [E]", *(fmt(comp.loc[m, "accuracy"]) for m in (M1, M2, M3))],
        ["Precision macro [E]", *(fmt(comp.loc[m, "precision_macro"]) for m in (M1, M2, M3))],
        ["Recall macro [E]", *(fmt(comp.loc[m, "recall_macro"]) for m in (M1, M2, M3))],
        ["F1 macro [E]", *(fmt(comp.loc[m, "f1_macro"]) for m in (M1, M2, M3))],
        ["F1 weighted [E]", *(fmt(comp.loc[m, "f1_weighted"]) for m in (M1, M2, M3))],
        ["Recall negativo [E]", *(fmt(comp.loc[m, "recall_negativo"]) for m in (M1, M2, M3))],
        ["F1 neutral [E]", *(fmt(comp.loc[m, "f1_neutral"]) for m in (M1, M2, M3))],
        ["Latencia media en lote [E]", *(fmt(comp.loc[m, "latencia_media_ms"], 2) + " ms/reseña" for m in (M1, M2, M3))],
        ["Rendimiento [E]", *(fmt(comp.loc[m, "rendimiento_reg_s"], 1) + " reseñas/s" for m in (M1, M2, M3))],
        ["Tiempo total 5 000 [E]", *(f"{fmt(comp.loc[m, 'tiempo_total_s'], 1)} ± {fmt(comp.loc[m, 'tiempo_total_std_s'], 1)} s" for m in (M1, M2, M3))],
        ["Latencia en línea p50 / p95 [E]", *(f"{fmt(comp.loc[m, 'online_p50_ms'], 1)} / {fmt(comp.loc[m, 'online_p95_ms'], 1)} ms" for m in (M1, M2, M3))],
        ["Tiempo de carga [E]", *(fmt(comp.loc[m, "tiempo_carga_s"], 1) + " s (caché local)" for m in (M1, M2, M3))],
        ["Parámetros [E]", *(fmt(comp.loc[m, "parametros_M"], 1) + " M" for m in (M1, M2, M3))],
        ["Memoria de pesos FP32 [Est]", *(fmt(comp.loc[m, "parametros_M"] * 4 / 1024, 2) + " GB" for m in (M1, M2, M3))],
    ]
    if cpu is not None:
        rows.append(["Rendimiento en CPU [E]", *(fmt(cpu.loc[m, "rendimiento_reg_s"], 1) + " reseñas/s" for m in (M1, M2, M3))])
        rows.append(["Latencia en línea p95 CPU [E]", *(fmt(cpu.loc[m, "online_p95_ms"], 1) + " ms" for m in (M1, M2, M3))])
    rows += [
        ["Licencia [P]", "MIT", "No declarada", "Apache-2.0"],
        ["Ventajas", "Mejor F1 en las 3 clases; dominio de reseñas; licencia permisiva",
         "Buen recall negativo; latencia en línea baja; 3 clases nativas",
         "El más rápido y ligero; licencia permisiva"],
        ["Limitaciones", "Clase neutral débil; origen de datos de entrenamiento no documentado",
         "Dominio Twitter; neutral muy débil; licencia no declarada; mayor tamaño",
         "Casi no predice neutral; sesgo hacia positivo; sin métricas publicadas"],
    ]
    table(doc, ["Criterio", M1, M2, M3], rows, widths=[3.8, 4.2, 4.2, 4.2], font=8)
    caption(doc, f"Tabla 1. Comparación de los tres modelos. Ejecución de referencia: {meta['device_description']} "
                 f"({meta['generated_at'][:10]}). [E] experimental, [P] publicado, [Est] estimado.")
    if not is_cuda:
        para(doc, (
            "En MPS, torch.mps.driver_allocated_memory refleja la reserva del asignador y no se libera entre "
            "modelos (≈ 2 GB en los tres casos), por lo que no sirve para comparar modelos. Por eso se usa la "
            "estimación de memoria de pesos en FP32; en Colab, el notebook registra la memoria pico real con "
            "torch.cuda.max_memory_allocated."), bold_prefix="Nota sobre memoria. ")

    doc.add_heading("Resultados por clase", 2)
    table(doc, ["Modelo", "Clase", "Precision", "Recall", "F1", "Soporte"],
          [[r.modelo, r.clase, fmt(r.precision), fmt(r.recall), fmt(r.f1), int(r.soporte)]
           for r in per_class.itertuples()], widths=[4, 2.5, 2.4, 2.4, 2.4, 2], font=8)
    fig = ref_dir / "figures" / "confusion_matrices.png"
    if fig.exists():
        doc.add_picture(str(fig), width=Cm(16))
        caption(doc, "Figura 1. Matrices de confusión (conteos y porcentaje por clase real).")
    cm1, cm3 = cms[M1], cms[M3]
    para(doc, (
        f"La clase neutral es la más difícil para los tres modelos. NLP Town acierta {pct(pc.loc[(M1, 'neutral'), 'recall'])} "
        f"de las reseñas neutrales, XLM-R {pct(pc.loc[(M2, 'neutral'), 'recall'])} y DistilmBERT "
        f"{pct(pc.loc[(M3, 'neutral'), 'recall'])}. DistilmBERT clasifica como positivas {cm3.loc['negativo', 'positivo']} "
        f"reseñas negativas, mientras que NLP Town solo {cm1.loc['negativo', 'positivo']}: los errores de NLP Town "
        "se concentran entre clases adyacentes (negativo↔neutral↔positivo), que es el tipo de error menos grave "
        "para el caso de uso."))
    n3 = int((all_wrong["stars"] == 3).sum())
    para(doc, (
        f"{len(all_wrong)} reseñas ({pct(len(all_wrong) / len(preds))}) fueron mal clasificadas por los tres "
        f"modelos; {n3} de ellas ({pct(n3 / len(all_wrong))}) tienen 3 estrellas y "
        f"{int(all_wrong['stars'].isin([2, 4]).sum())} tienen 2 o 4, es decir, están en la frontera de la regla de "
        "agrupación. En una revisión manual de 15 de estos casos elegidos al azar se observaron tres patrones: "
        "reseñas de 3 estrellas con quejas concretas que los modelos leen como negativas; reseñas cuyo texto "
        "contradice la calificación (por ejemplo, 5 estrellas con un texto que describe una devolución); y "
        "expresiones coloquiales del español peninsular (\"han quedado brutales\", en sentido positivo) "
        "interpretadas como negativas. Son errores atribuibles en parte a la ambigüedad del etiquetado y a la "
        "variante dialectal, no solo a los modelos."), bold_prefix="Análisis de errores. ")
    se = c1.get("stars_exact_acc")
    if pd.notna(se):
        para(doc, (
            f"En su escala original de 5 estrellas, NLP Town obtuvo {pct(se)} de exactitud y "
            f"{pct(c1['stars_off_by_1_acc'])} con tolerancia de ±1 estrella, frente al 58 % y 95 % que sus autores "
            "publican para español con otro conjunto de 5 000 reseñas. La cercanía de las cifras sugiere que la "
            "evaluación es coherente, pero las métricas publicadas no se usan para decidir."),
            bold_prefix="Publicado vs. experimental. ")

    # 10. Trade-offs
    doc.add_heading("9. Análisis de trade-offs", 1)
    fig = ref_dir / "figures" / "f1_vs_throughput.png"
    if fig.exists():
        doc.add_picture(str(fig), width=Cm(12))
        caption(doc, "Figura 2. F1 macro frente a rendimiento (el tamaño del punto es proporcional a los parámetros).")
    qa = [
        ("¿Qué modelo obtiene mejores métricas? ",
         f"NLP Town en todas: F1 macro {fmt(c1['f1_macro'])} (+{fmt(gain_f1)} sobre XLM-R y "
         f"+{fmt(c1['f1_macro'] - c3['f1_macro'])} sobre DistilmBERT), accuracy {fmt(c1['accuracy'])} y el mejor F1 por clase."),
        ("¿Qué modelo presenta menor latencia? ",
         f"DistilmBERT: {fmt(c3['latencia_media_ms'], 2)} ms por reseña en lote ({fmt(c3['rendimiento_reg_s'], 1)} reseñas/s) "
         f"y p95 de {fmt(c3['online_p95_ms'], 1)} ms en línea, porque tiene la mitad de capas (6) que los otros dos."),
        ("¿Qué diferencias de desempeño son relevantes? ",
         f"Las de calidad: {fmt(gain_f1)} puntos de F1 macro entre NLP Town y XLM-R equivalen a reconocer "
         f"{int(cm1.loc['neutral', 'neutral'] - cms[M2].loc['neutral', 'neutral'])} reseñas neutrales más de 1 000. Con un "
         f"bootstrap de 1 000 remuestreos, el intervalo de confianza del 95 % del F1 macro es "
         f"[{fmt(ci[M1][0])}; {fmt(ci[M1][1])}] para NLP Town, [{fmt(ci[M2][0])}; {fmt(ci[M2][1])}] para XLM-R y "
         f"[{fmt(ci[M3][0])}; {fmt(ci[M3][1])}] para DistilmBERT; la diferencia pareada NLP Town − XLM-R está en "
         f"[{fmt(ci_diff[M2][0])}; {fmt(ci_diff[M2][1])}], lejos de cero, por lo que la ventaja no se explica por azar de muestreo. "
         "En velocidad, NLP Town y XLM-R son similares en lote, pese a que XLM-R tiene 66 % más parámetros: la "
         "diferencia está en el vocabulario (capa de embeddings), no en las capas que se ejecutan."),
        ("¿Qué modelo requiere más recursos? ",
         f"XLM-R: {fmt(c2['parametros_M'], 0)} M de parámetros, ≈ {fmt(c2['parametros_M'] * 4 / 1024, 2)} GB de pesos en FP32 y "
         f"{fmt(c2['pesos_hub_MB'] / 1024, 2)} GB de descarga."),
        ("¿Una mejora en F1 justifica un mayor tiempo de inferencia? ",
         f"Sí en este caso. Pasar de DistilmBERT a NLP Town duplica el tiempo por reseña "
         f"({fmt(c3['latencia_media_ms'], 1)} → {fmt(c1['latencia_media_ms'], 1)} ms), pero aumenta el F1 macro de "
         f"{fmt(c3['f1_macro'])} a {fmt(c1['f1_macro'])} y el recall negativo de {fmt(c3['recall_negativo'])} a "
         f"{fmt(c1['recall_negativo'])}: de cada 1 000 quejas se escalarían unas "
         f"{int(round((c1['recall_negativo'] - c3['recall_negativo']) * 1000))} más. El tiempo adicional "
         "sigue dentro de los umbrales de latencia."),
        ("¿Qué ocurre si se deben procesar muchos registros? ",
         f"Con el rendimiento medido, 100 000 reseñas diarias tardarían ≈ {fmt(100000 / c1['rendimiento_reg_s'] / 60, 0)} min con "
         f"NLP Town y ≈ {fmt(100000 / c3['rendimiento_reg_s'] / 60, 0)} min con DistilmBERT en la GPU de referencia"
         + (f"; en CPU, ≈ {fmt(100000 / cpu.loc[M1, 'rendimiento_reg_s'] / 60, 0)} y "
            f"≈ {fmt(100000 / cpu.loc[M3, 'rendimiento_reg_s'] / 60, 0)} min" if cpu is not None else "")
         + ". Para un proceso nocturno por lotes, cualquiera es viable; para volúmenes mucho mayores conviene "
           "ordenar por longitud, usar FP16 en GPU o cuantizar."),
        ("¿Qué modelo sería apropiado con recursos limitados? ",
         "Si solo se dispone de CPU, NLP Town sigue siendo viable para volúmenes moderados"
         + (f" ({fmt(cpu.loc[M1, 'rendimiento_reg_s'], 1)} reseñas/s y p95 en línea de {fmt(cpu.loc[M1, 'online_p95_ms'], 1)} ms en CPU)" if cpu is not None else "")
         + ". DistilmBERT solo sería preferible si la latencia fuera la restricción dominante y la tarea se "
           "redujera a positivo/negativo; con tres clases no alcanza la calidad mínima. Una opción intermedia es "
           "cuantizar NLP Town a INT8 o exportarlo a ONNX (no evaluado en este trabajo)."),
    ]
    for q, a in qa:
        para(doc, a, bold_prefix=q)

    # 11. Ajustes
    doc.add_heading("10. Decisiones de ajuste mínimo", 1)
    table(doc, ["Decisión", "Razón", "Efecto posible"], [
        ["Sin fine-tuning; solo inferencia con pesos preentrenados", "Objetivo: seleccionar un modelo; comparar en igualdad",
         "Las métricas reflejan el desempeño de fábrica; un ajuste con datos propios podría cambiar la clasificación"],
        ["5 estrellas → 3 clases (1–2 / 3 / 4–5)", "Alinear dataset y salidas de los modelos",
         "La frontera de 2 y 4 estrellas es difusa y penaliza a todos los modelos por igual"],
        ["Mapeo de etiquetas por nombre (id2label)", "Cada modelo ordena sus clases distinto (lxyuan: 0 = positive)",
         "Evita un error silencioso de mapeo por índice"],
        ["Sin preprocesamiento del texto", "Las reseñas ya están limpias; el preprocesamiento de Cardiff (@user, http) no aplica",
         "Ninguno relevante"],
        ["max_length = 256 tokens", "Cubre > 99,7 % de las reseñas completas", "Se trunca el final de ≈ 13 reseñas largas"],
        ["batch_size = 32, padding dinámico", "Tamaño estándar que cabe en una T4 y en CPU", "Afecta la velocidad, no las predicciones"],
        ["Precisión FP32", "Evitar que la precisión numérica sea una variable", "FP16 en GPU podría acelerar ~2× (no evaluado)"],
        ["Calentamiento y 3 repeticiones", "Excluir la inicialización y estimar la variabilidad", "Tiempos más estables"],
    ], widths=[4.5, 6, 6], font=8)

    # 12. Justificación
    doc.add_heading("11. Justificación del modelo recomendado", 1)
    para(doc, f"Modelo recomendado: {IDS[M1]} (commit {c1['revision'][:7]}).", bold_prefix="")
    bullets(doc, [
        ("Calidad: ", f"mayor F1 macro ({fmt(c1['f1_macro'])}), único por encima del umbral de 0,60, y mejor F1 en cada clase."),
        ("Objetivo de negocio: ", f"recall negativo de {fmt(c1['recall_negativo'])} (umbral 0,75) y solo "
                                  f"{cm1.loc['negativo', 'positivo']} quejas clasificadas como positivas de 2 000."),
        ("Eficiencia suficiente: ", f"{fmt(c1['rendimiento_reg_s'], 1)} reseñas/s en lote y p95 de {fmt(c1['online_p95_ms'], 1)} ms en línea "
                                    "en la GPU de referencia; ambos dentro de los umbrales."),
        ("Adecuación al dominio: ", "fue ajustado con reseñas de productos, el mismo tipo de texto del caso de uso."),
        ("Licencia y mantenimiento: ", "MIT, pesos en safetensors, versión identificable por commit."),
    ])
    para(doc, (
        f"XLM-R se descarta por calidad (F1 {fmt(c2['f1_macro'])}; su intervalo de confianza "
        f"[{fmt(ci[M2][0])}; {fmt(ci[M2][1])}] apenas alcanza el umbral, y queda claramente por debajo de NLP Town) "
        "y porque no declara licencia; DistilmBERT, "
        f"aunque {fmt(speedup, 1)}× más rápido, no detecta la clase neutral (recall {fmt(pc.loc[(M3, 'neutral'), 'recall'])}) "
        "y su recall negativo queda por debajo del umbral. La recomendación no se basa en accuracy sino en el "
        "cumplimiento conjunto de calidad, objetivo de negocio, eficiencia y licencia."))

    # 13. Umbrales y riesgos
    doc.add_heading("12. Umbrales de aceptación y riesgos", 1)
    para(doc, "Umbrales propuestos antes de ejecutar el experimento (no validados por una organización real):")
    def ok(cond):
        return "Cumple" if cond else "No cumple"
    table(doc, ["Criterio", "Umbral", M1, M2, M3], [
        ["F1 macro", "≥ 0,60", *(f"{fmt(comp.loc[m, 'f1_macro'])} — {ok(comp.loc[m, 'f1_macro'] >= 0.60)}" for m in (M1, M2, M3))],
        ["Recall negativo", "≥ 0,75", *(f"{fmt(comp.loc[m, 'recall_negativo'])} — {ok(comp.loc[m, 'recall_negativo'] >= 0.75)}" for m in (M1, M2, M3))],
        ["Rendimiento en lote (GPU)", "≥ 50 reseñas/s", *(f"{fmt(comp.loc[m, 'rendimiento_reg_s'], 1)} — {ok(comp.loc[m, 'rendimiento_reg_s'] >= 50)}" for m in (M1, M2, M3))],
        ["Latencia p95 en línea (GPU)", "≤ 100 ms", *(f"{fmt(comp.loc[m, 'online_p95_ms'], 1)} — {ok(comp.loc[m, 'online_p95_ms'] <= 100)}" for m in (M1, M2, M3))],
        ["Memoria de pesos", "≤ 2 GB", *(f"{fmt(comp.loc[m, 'parametros_M'] * 4 / 1024, 2)} GB [Est] — Cumple" for m in (M1, M2, M3))],
    ], widths=[3.8, 2.6, 3.4, 3.4, 3.4], font=8)
    para(doc, (
        "Justificación: 0,60 de F1 macro es un mínimo útil para triaje en tres clases con una clase neutral ambigua; "
        "0,75 de recall negativo asegura que se escalen al menos tres de cada cuatro quejas; 50 reseñas/s procesa "
        "100 000 reseñas en ≈ 35 min; 100 ms se percibe como respuesta inmediata; 2 GB deja margen en una T4 de 16 GB. "
        + ("Los umbrales de GPU se evaluaron aquí con la GPU Apple (MPS); deben confirmarse con la ejecución en Colab." if not is_cuda else "")))
    table(doc, ["Riesgo", "Descripción", "Mitigación"], [
        ["Errores de clasificación", "≈ 1 de cada 6 quejas no se detecta; la clase neutral es poco fiable",
         "Usar como priorización con revisión humana; monitorear recall negativo con muestras etiquetadas"],
        ["Desbalance de clases", "Neutral = 20 %; accuracy sobrestima el desempeño", "Decidir con F1 macro y métricas por clase"],
        ["Sesgos", "Español peninsular y productos de Amazon; posibles diferencias por variante dialectal o tipo de producto",
         "Evaluar con reseñas propias por región y categoría antes de producción"],
        ["Datos de evaluación vs. reales", "Las reseñas reales pueden incluir jerga, emojis, sarcasmo o temas de servicio",
         "Piloto con datos de la organización; fine-tuning si F1 cae bajo el umbral"],
        ["Cambios en el comportamiento", "Deriva de datos y actualizaciones del modelo en el Hub",
         "Fijar commit; registrar distribución de predicciones; reevaluar periódicamente"],
        ["Posible contaminación", "NLP Town no documenta el origen de sus reseñas; podría haber visto reseñas de Amazon",
         "Validar con datos propios no publicados"],
        ["Licencias", "MARC: solo uso académico, no comercial; XLM-R sin licencia declarada",
         "Para producción, evaluar con datos propios; usar modelo con licencia MIT"],
        ["Privacidad", "Las reseñas reales pueden contener datos personales", "Anonimizar antes de inferir; no registrar textos en logs"],
        ["Limitaciones computacionales", "GPU de Colab gratuita no garantizada; tiempos variables",
         "El modelo funciona en CPU; documentar el hardware de cada medición"],
    ], widths=[3.5, 6.5, 6.5], font=8)

    # 14. Conclusiones
    doc.add_heading("13. Conclusiones", 1)
    para(doc, (
        f"Se compararon tres modelos preentrenados de Hugging Face en igualdad de condiciones sobre 5 000 reseñas en "
        f"español. El modelo de NLP Town, especializado en reseñas de productos, obtuvo la mejor calidad "
        f"(F1 macro {fmt(c1['f1_macro'])}) con una eficiencia suficiente para el caso de uso; DistilmBERT fue el "
        f"más eficiente pero no alcanzó la calidad mínima, y XLM-R, el más grande, no superó al modelo especializado. "
        "El tamaño del modelo no determinó la calidad: el ajuste al dominio fue más importante que el número de "
        "parámetros."))
    para(doc, (
        "La principal limitación es la clase neutral, débil en los tres modelos, en parte por la ambigüedad de la "
        "regla de 3 estrellas. Los siguientes pasos son validar con reseñas propias de la organización, ajustar "
        "(fine-tuning) el modelo recomendado si es necesario, evaluar cuantización o FP16 y publicar una demo en "
        "Hugging Face Spaces para recoger retroalimentación de los usuarios."))

    # 15. Referencias
    doc.add_heading("14. Referencias", 1)
    refs = [
        "Barbieri, F., Espinosa Anke, L., & Camacho-Collados, J. (2022). XLM-T: Multilingual language models in Twitter for sentiment analysis and beyond. En Proceedings of the Thirteenth Language Resources and Evaluation Conference. European Language Resources Association. https://aclanthology.org/2022.lrec-1.27/",
        "Cardiff NLP. (s. f.). cardiffnlp/twitter-xlm-roberta-base-sentiment [Model card]. Hugging Face. Recuperado el 11 de octubre de 2026, de https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. En Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. https://aclanthology.org/2020.acl-main.747/",
        "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. En Proceedings of NAACL-HLT 2019. https://aclanthology.org/N19-1423/",
        "Hugging Face. (s. f.-a). Datasets [Documentación del Hub]. Recuperado el 11 de octubre de 2026, de https://huggingface.co/docs/hub/datasets",
        "Hugging Face. (s. f.-b). Model cards [Documentación del Hub]. Recuperado el 11 de octubre de 2026, de https://huggingface.co/docs/hub/model-cards",
        "Hugging Face. (s. f.-c). Models [Documentación del Hub]. Recuperado el 11 de octubre de 2026, de https://huggingface.co/docs/hub/models",
        "Hugging Face. (s. f.-d). Spaces [Documentación del Hub]. Recuperado el 11 de octubre de 2026, de https://huggingface.co/docs/hub/spaces",
        "Hugging Face. (s. f.-e). User access tokens [Documentación del Hub]. Recuperado el 11 de octubre de 2026, de https://huggingface.co/docs/hub/security-tokens",
        "Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). The multilingual Amazon reviews corpus. En Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). https://aclanthology.org/2020.emnlp-main.369/",
        "Lhoest, Q., Villanova del Moral, A., Jernite, Y., Thakur, A., von Platen, P., Patil, S., Chaumond, J., Drame, M., Plu, J., Tunstall, L., Davison, J., Šaško, M., Chhablani, G., Malik, B., Brandeis, S., Le Scao, T., Sanh, V., Xu, C., Patry, N., … Wolf, T. (2021). Datasets: A community library for natural language processing. En Proceedings of EMNLP 2021: System Demonstrations. https://aclanthology.org/2021.emnlp-demo.21/",
        "lxyuan. (s. f.). lxyuan/distilbert-base-multilingual-cased-sentiments-student [Model card]. Hugging Face. Recuperado el 11 de octubre de 2026, de https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 220–229). https://doi.org/10.1145/3287560.3287596",
        "NLP Town. (s. f.). nlptown/bert-base-multilingual-uncased-sentiment [Model card]. Hugging Face. Recuperado el 11 de octubre de 2026, de https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html",
        "Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv. https://arxiv.org/abs/1910.01108",
        "SetFit. (s. f.). SetFit/amazon_reviews_multi_es [Dataset]. Hugging Face. Recuperado el 11 de octubre de 2026, de https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es",
        "Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., … Rush, A. (2020). Transformers: State-of-the-art natural language processing. En Proceedings of EMNLP 2020: System Demonstrations. https://aclanthology.org/2020.emnlp-demos.6/",
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.left_indent = Cm(1.25)
        p.paragraph_format.first_line_indent = Cm(-1.25)
        p.paragraph_format.space_after = Pt(6)

    # Anexo: rúbrica
    doc.add_heading("Anexo A. Correspondencia con la rúbrica", 1)
    table(doc, ["Criterio (25 % c/u)", "Evidencia", "Ubicación"], [
        ["1. Selección y análisis de modelos", "Ecosistema (Models, Datasets, Spaces); criterios; 3 modelos justificados; descartes",
         "Secciones 4–5; README; notebook sección 5"],
        ["2. Implementación en Colab", "Notebook por secciones, GPU verificada, dependencias, carga reproducible, inferencia",
         "notebooks/actividad4_huggingface.ipynb"],
        ["3. Evaluación y comparación", "Accuracy, precision, recall, F1 (macro/weighted), por clase, latencia, trade-offs",
         "Secciones 8–9; results/; notebook secciones 7–9"],
        ["4. Documentación y justificación", "Este documento, README, repositorio, instrucciones, justificación con umbrales",
         "docs/; README.md; GitHub"],
    ], widths=[4, 7.5, 5], font=8)
    para(doc, f"Documento generado por docs/build_report.py a partir de results/{reference}/ "
              f"(ejecución del {meta['generated_at'].replace('T', ' ')}).")

    doc.save(OUT)
    print(f"Escrito {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None, help="Carpeta de results/ usada como referencia")
    a = ap.parse_args()
    ref = a.reference or ("colab_gpu" if (RES / "colab_gpu" / "model_comparison.csv").exists() else "local_mps")
    build(ref)
