# Selección de un modelo preentrenado de Hugging Face para clasificar el sentimiento de reseñas en español

Actividad 4 — *Gestión de proyectos de inteligencia artificial* (Universidad Tecmilenio).

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marlim21/actividad4-huggingface/blob/main/notebooks/actividad4_huggingface.ipynb)

Comparación experimental de **tres modelos preentrenados del Hugging Face Hub** que clasifican reseñas de clientes en español como **negativo / neutral / positivo**. Se mide la calidad predictiva (accuracy, precision, recall, F1), la latencia, el rendimiento y los recursos en condiciones equivalentes, y se recomienda un modelo justificado por métricas y objetivos de negocio.

> Solo se hace **inferencia con pesos preentrenados**: ningún modelo fue ajustado (*fine-tuning*) ni entrenado.

## 1. Problema y objetivos

**Problema.** Una empresa de comercio electrónico recibe miles de reseñas en texto libre. Leerlas manualmente retrasa la atención de quejas y no permite medir la satisfacción de forma continua.

**Objetivo general.** Identificar, implementar, evaluar y seleccionar el modelo preentrenado más adecuado para clasificar el sentimiento de reseñas en español, equilibrando calidad, latencia y recursos.

**Objetivos específicos.**
1. Definir criterios de selección y elegir tres modelos candidatos.
2. Preparar un conjunto de prueba reproducible.
3. Implementar la inferencia en Google Colab con GPU.
4. Medir accuracy, precision, recall, F1, latencia y recursos en condiciones equivalentes.
5. Recomendar un modelo justificado frente a umbrales de aceptación definidos de antemano.

**Usuarios.** Agentes de servicio al cliente (cola de reseñas negativas), analistas de producto y gerencia (indicadores de satisfacción).

## 2. Tecnologías

Python · PyTorch · Hugging Face Transformers, Datasets y Hub · scikit-learn · pandas · matplotlib · Google Colab (GPU T4) · python-docx (solo para generar el informe).

## 3. Modelos evaluados

| # | Modelo (commit fijado) | Arquitectura | Salida original | Parámetros | Licencia |
|---|---|---|---|---|---|
| 1 | [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) (`8f6f4e3`) | BERT-base multilingüe, 12 capas | 1–5 estrellas (reseñas de productos) | 167,4 M | MIT |
| 2 | [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) (`f2f1202`) | XLM-RoBERTa-base, 12 capas | negative / neutral / positive (tuits) | 278,0 M | No declarada |
| 3 | [`lxyuan/distilbert-base-multilingual-cased-sentiments-student`](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student) (`cf99110`) | DistilBERT multilingüe, 6 capas | positive / neutral / negative | 135,3 M | Apache-2.0 |

Las etiquetas de cada modelo se normalizan **por nombre** (`config.id2label`) a las tres clases objetivo. Para el modelo 1 se aplica la misma regla que al dataset: 1–2★ = negativo, 3★ = neutral, 4–5★ = positivo. Los criterios de selección y los candidatos descartados se documentan en [docs/analisis_comparativo.docx](docs/analisis_comparativo.docx).

## 4. Dataset

Partición en español del **Multilingual Amazon Reviews Corpus** (Keung et al., 2020), tomada de la réplica [`SetFit/amazon_reviews_multi_es`](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es) (commit `1601541`). Se evalúa la **partición `test` oficial completa: 5 000 reseñas** (negativo 2 000 · neutral 1 000 · positivo 2 000).

⚠️ La licencia del corpus permite solo uso académico y **prohíbe republicarlo**; por eso el CSV con textos se genera localmente y no está en el repositorio. Detalles, diccionario de datos y limitaciones: [data/README.md](data/README.md).

## 5. Estructura del repositorio

```
actividad4-huggingface/
├── README.md
├── requirements.txt
├── LICENSE                         # MIT, solo para el código
├── .gitignore
├── notebooks/
│   ├── actividad4_huggingface.ipynb   # notebook para Google Colab (GPU), sin salidas
│   └── evidencia/
│       └── actividad4_huggingface_ejecutado_local_mps.ipynb  # misma versión, ejecutada en local con salidas
├── src/
│   ├── utils.py                    # configuración común: dataset, modelos, mapeo de etiquetas, umbrales
│   ├── prepare_data.py             # descarga, validación y transformación de datos
│   ├── evaluate_models.py          # inferencia, métricas, latencia y recursos
│   └── consolidate_results.py      # une las ejecuciones en results/model_comparison.csv
├── data/
│   ├── README.md                   # fuente, licencia, diccionario de datos
│   └── processed/
│       ├── dataset_manifest.csv    # id, estrellas, sentimiento (sin texto)
│       ├── data_profile.json       # estadísticas de preparación
│       └── dataset.csv             # se genera localmente (no se publica)
├── results/
│   ├── model_comparison.csv        # tabla consolidada de todas las ejecuciones
│   ├── local_mps/                  # ejecución de referencia: GPU Apple M2 Pro (MPS)
│   ├── local_cpu/                  # misma evaluación en CPU
│   ├── notebook_local/             # resultados del notebook ejecutado en local (validación)
│   └── colab_gpu/                  # se agrega al ejecutar el notebook en Colab (pendiente)
└── docs/
    ├── analisis_comparativo.docx   # informe técnico
    └── build_report.py             # genera el informe a partir de results/
```

Cada carpeta de `results/<ejecución>/` contiene `model_comparison.csv`, `per_class_metrics.csv`, `predictions.csv` (sin texto), `confusion_<modelo>.csv`, `run_metadata.json` y `figures/`.

## 6. Requisitos

- Python ≥ 3.10 (probado con 3.13 en macOS; Colab usa su propia versión).
- ~3 GB libres para la caché de modelos de Hugging Face.
- GPU opcional: CUDA (Colab) o MPS (Apple Silicon). En CPU funciona, más lento.
- No se necesita token de Hugging Face: todos los recursos son públicos.

## 7. Ejecución en Google Colab

1. Abra el notebook con el botón **Abrir en Colab** de arriba (o `Archivo → Subir notebook`).
2. `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)`.
3. `Entorno de ejecución → Ejecutar todas`. Tarda de 5 a 10 minutos incluyendo descargas.
4. La sección 2 imprime la GPU detectada. Si no hay GPU, el notebook continúa en CPU y lo advierte.
5. La última celda de la sección 9 descarga `colab_gpu_results.zip`. Descomprímalo en `results/colab_gpu/` y regenere la tabla y el informe:

```bash
python src/consolidate_results.py
```

```bash
python docs/build_report.py --reference colab_gpu
```

**Autenticación (opcional).** Si en el futuro se usan recursos privados, cree un token de solo lectura, guárdelo en el panel 🔑 *Secretos* de Colab como `HF_TOKEN` y el notebook lo usará sin mostrarlo. Nunca pegue tokens en celdas ni los suba a GitHub.

## 8. Ejecución local de los scripts

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python src/prepare_data.py
```

```bash
python src/evaluate_models.py --out results/local_mps
```

```bash
python src/evaluate_models.py --device cpu --out results/local_cpu
```

```bash
python src/consolidate_results.py
```

```bash
python docs/build_report.py
```

`evaluate_models.py --limit 300 --repeats 1` hace una prueba rápida. `--device` acepta `auto`, `cuda`, `mps` o `cpu`.

**Validación.** `prepare_data.py` falla si faltan columnas, hay valores faltantes, IDs repetidos o etiquetas fuera de rango. `evaluate_models.py` falla si un modelo emite una etiqueta sin mapeo y avisa si las predicciones difieren entre repeticiones. `build_report.py` se detiene si las afirmaciones del informe dejan de ser ciertas con los resultados.

## 9. Métricas

- **Accuracy**, **precision**, **recall** y **F1** (scikit-learn), con promedio **macro** y **weighted**, reporte por clase y matriz de confusión.
- **Métrica principal: F1 macro**, porque las clases están desbalanceadas (20 % neutral) y el caso de uso necesita las tres clases.
- **Métrica de negocio: recall de la clase negativa** (quejas que sí se escalan).
- **Eficiencia:** tiempo de carga; tiempo total, latencia media y rendimiento en lote (batch 32, 3 lotes de calentamiento, 3 repeticiones); latencia en línea con batch 1 (media, p50, p95; 200 reseñas); parámetros, tamaño de pesos y memoria.

## 10. Resultados experimentales

Partición `test` completa (5 000 reseñas), FP32, `batch_size=32`, `max_length=256`. Ejecutado el 11-oct-2026 en un Apple M2 Pro (16 GB) con PyTorch 2.14.1 y Transformers 5.19.0.

| Modelo | Accuracy | Precision macro | Recall macro | **F1 macro** | F1 weighted | Recall negativo | Rendimiento MPS (reseñas/s) | p95 en línea MPS (ms) | Rendimiento CPU (reseñas/s) |
|---|---|---|---|---|---|---|---|---|---|
| mBERT-nlptown | 0,766 | 0,717 | 0,716 | **0,716** | 0,765 | 0,842 | 98,4 | 40,9 | 46,7 |
| XLM-R-cardiffnlp | 0,675 | 0,593 | 0,595 | 0,588 | 0,659 | 0,828 | 107,6 | 18,5 | 48,3 |
| DistilmBERT-lxyuan | 0,632 | 0,524 | 0,541 | 0,514 | 0,592 | 0,668 | 204,5 | 13,2 | 90,1 |

- Las métricas de calidad fueron **idénticas en MPS y en CPU**; los tiempos dependen del dispositivo.
- IC 95 % del F1 macro (bootstrap, 1 000 remuestreos): nlptown [0,702; 0,728] · XLM-R [0,575; 0,601] · DistilmBERT [0,502; 0,526]. Diferencia pareada nlptown − XLM-R: [0,113; 0,144].
- F1 de la clase neutral: 0,471 (nlptown), 0,233 (XLM-R), 0,127 (DistilmBERT).
- En su escala de 5 estrellas, nlptown obtuvo 56,5 % de exactitud y 93,1 % con ±1 estrella (la *model card* publica 58 % y 95 % para español con otros datos).
- La memoria reportada por MPS no distingue entre modelos (≈ 2 GB en los tres casos, reserva del asignador); el informe usa la memoria de pesos estimada en FP32 (0,65 / 1,09 / 0,53 GB). En Colab se registra la memoria pico real de CUDA.
- **Pendiente:** ejecución del notebook en Google Colab con GPU CUDA (`results/colab_gpu/`). Los tiempos de esta tabla **no** son de Colab.

Figuras: [matrices de confusión](results/local_mps/figures/confusion_matrices.png) · [F1 vs. rendimiento](results/local_mps/figures/f1_vs_throughput.png).

## 11. Conclusiones

- **Mejor desempeño predictivo:** `nlptown/bert-base-multilingual-uncased-sentiment` (F1 macro 0,716; mejor en las tres clases).
- **Mejor eficiencia:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student` (≈ 2× más rápido), pero casi no reconoce la clase neutral.
- **Mejor equilibrio y recomendado:** `nlptown/bert-base-multilingual-uncased-sentiment`. Es el único que cumple todos los umbrales propuestos (F1 macro ≥ 0,60; recall negativo ≥ 0,75; ≥ 50 reseñas/s y p95 ≤ 100 ms en GPU), está ajustado al dominio de reseñas y tiene licencia MIT.
- El tamaño no determinó la calidad: XLM-R tiene 66 % más parámetros (sobre todo en su vocabulario) y quedó por debajo del modelo especializado.

## 12. Limitaciones

- Solo inferencia; un ajuste con datos propios podría cambiar el orden de los modelos.
- Reseñas de amazon.es (español peninsular, solo productos); pueden no representar a la organización destino.
- La regla 3★ = neutral es una convención; la clase neutral es ambigua para los tres modelos.
- NLP Town no documenta el origen de sus reseñas de entrenamiento: posible contaminación o ventaja de dominio.
- Los umbrales de aceptación son propuestos, no validados por una organización real.
- Los tiempos de GPU se midieron en Apple MPS; falta la medición en Colab con CUDA.

## 13. Estado de validación y cumplimiento de la rúbrica

| Criterio (25 % c/u) | Evidencia | Ubicación | Estado |
|---|---|---|---|
| 1. Selección y análisis de modelos | Ecosistema HF (Models, Datasets, Spaces), criterios, 3 modelos justificados, descartes | Informe §4–5, este README §3 | Documentado y verificado contra el Hub |
| 2. Implementación en Colab | Notebook por secciones, verificación de GPU, dependencias, carga reproducible, inferencia | `notebooks/` | Ejecutado y validado **en local (MPS)** sin errores; **pendiente la ejecución en Colab con GPU** |
| 3. Evaluación y comparación | Accuracy, precision, recall, F1 macro/weighted, por clase, IC bootstrap, latencia, recursos, trade-offs | `results/`, informe §8–9 | Ejecutado y validado (scripts y notebook dan predicciones idénticas) |
| 4. Documentación y justificación | Informe Word, README, scripts, repositorio | `docs/`, `README.md`, GitHub | Generado; publicación en GitHub pendiente |

Qué se ejecutó realmente: `prepare_data.py` y `evaluate_models.py` (MPS y CPU, 5 000 reseñas, 3 repeticiones), el notebook completo en local (MPS) y la instalación de `requirements.txt` en un entorno limpio.

## 14. Licencias y condiciones de uso

| Recurso | Licencia | Implicación |
|---|---|---|
| Código de este repositorio | MIT ([LICENSE](LICENSE)) | Reutilizable con atribución |
| Multilingual Amazon Reviews Corpus | Licencia propia de Amazon: solo investigación académica, sin uso comercial ni republicación | No se publica el texto; se regenera con `prepare_data.py` |
| `nlptown/bert-base-multilingual-uncased-sentiment` | MIT | Uso permitido con atribución |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | No declarada en la *model card* | Uso académico; revisar antes de un uso comercial |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | Apache-2.0 | Uso permitido con atribución y aviso de licencia |

Este repositorio no redistribuye pesos de modelos ni el texto del corpus.

## 15. Reproducibilidad

Semilla 42 · commits fijados de dataset y modelos en [src/utils.py](src/utils.py) · versiones exactas en `results/*/run_metadata.json` · dependencias en [requirements.txt](requirements.txt). En Colab: `Entorno de ejecución → Desconectar y borrar el entorno de ejecución` y `Ejecutar todas`.

## 16. Actualizar el repositorio

```bash
git status
```

```bash
git add -A
```

```bash
git commit -m "Describe el cambio"
```

```bash
git push
```

## 17. Referencias

- Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). The multilingual Amazon reviews corpus. *EMNLP 2020*. https://aclanthology.org/2020.emnlp-main.369/
- Barbieri, F., Espinosa Anke, L., & Camacho-Collados, J. (2022). XLM-T: Multilingual language models in Twitter for sentiment analysis and beyond. *LREC 2022*. https://aclanthology.org/2022.lrec-1.27/
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT. *NAACL-HLT 2019*. https://aclanthology.org/N19-1423/
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*. https://aclanthology.org/2020.acl-main.747/
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT. arXiv. https://arxiv.org/abs/1910.01108
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020 Demos*. https://aclanthology.org/2020.emnlp-demos.6/
- Lhoest, Q., et al. (2021). Datasets: A community library for natural language processing. *EMNLP 2021 Demos*. https://aclanthology.org/2021.emnlp-demo.21/
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Hugging Face. (s. f.). Documentación del Hub: [Models](https://huggingface.co/docs/hub/models), [Datasets](https://huggingface.co/docs/hub/datasets), [Spaces](https://huggingface.co/docs/hub/spaces), [Model cards](https://huggingface.co/docs/hub/model-cards).

Lista completa en formato APA 7 en el informe Word.
