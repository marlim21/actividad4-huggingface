"""Funciones y configuración compartidas por prepare_data.py y evaluate_models.py.

Toda la configuración experimental (dataset, revisiones de modelos, mapeo de
etiquetas, semilla y umbrales de aceptación) vive aquí para que ambos scripts y
el notebook usen exactamente los mismos valores.
"""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Configuración experimental
# --------------------------------------------------------------------------
SEED = 42

# Réplica en el Hub de la partición en español del Multilingual Amazon Reviews
# Corpus (Keung et al., 2020). Se fija el commit para que los datos no cambien.
DATASET_ID = "SetFit/amazon_reviews_multi_es"
DATASET_REVISION = "16015418b488c9186fce74b058877ea939ca934d"
DATASET_SPLIT = "test"  # partición de prueba oficial (5 000 reseñas)

# Modelos evaluados: identificador del Hub -> commit fijado.
MODELS = {
    "nlptown/bert-base-multilingual-uncased-sentiment": "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student": "cf991100d706c13c0a080c097134c05b7f436c45",
}
SHORT_NAMES = {
    "nlptown/bert-base-multilingual-uncased-sentiment": "mBERT-nlptown",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": "XLM-R-cardiffnlp",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student": "DistilmBERT-lxyuan",
}

# Clases objetivo del caso de uso (3 clases).
LABELS = ["negativo", "neutral", "positivo"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def stars_to_label(stars: int) -> str:
    """Estrellas (1-5) -> sentimiento. 1-2 negativo, 3 neutral, 4-5 positivo."""
    if stars <= 2:
        return "negativo"
    if stars == 3:
        return "neutral"
    return "positivo"


# Normalización de las etiquetas que emite cada modelo (en minúsculas) hacia
# las 3 clases objetivo. nlptown predice estrellas, por lo que se aplica la
# misma regla que a las etiquetas reales del dataset.
MODEL_LABEL_MAP = {
    "1 star": "negativo", "2 stars": "negativo", "3 stars": "neutral",
    "4 stars": "positivo", "5 stars": "positivo",
    "negative": "negativo", "neutral": "neutral", "positive": "positivo",
}


def normalize_model_label(raw_label: str) -> str:
    key = raw_label.strip().lower()
    if key not in MODEL_LABEL_MAP:
        raise ValueError(f"Etiqueta de modelo sin mapeo: {raw_label!r}")
    return MODEL_LABEL_MAP[key]


# Condiciones de inferencia idénticas para los tres modelos.
INFERENCE = {
    "batch_size": 32,
    "max_length": 256,      # tokens; ver data_profile.json para el % truncado
    "warmup_batches": 3,
    "timed_repeats": 3,     # pasadas completas cronometradas sobre la prueba
    "online_samples": 200,  # reseñas procesadas una a una (batch_size=1)
}

# Umbrales de aceptación PROPUESTOS (definidos antes de ejecutar el experimento;
# no han sido validados por una organización real).
ACCEPTANCE = {
    "f1_macro_min": 0.60,
    "recall_negativo_min": 0.75,
    "throughput_gpu_min": 50.0,     # reseñas/s en lote con GPU
    "online_p95_ms_gpu_max": 100.0, # latencia p95 por reseña (batch_size=1) con GPU
    "peak_memory_gb_max": 2.0,
}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto"):
    """Devuelve torch.device. 'auto' prioriza CUDA, luego MPS (Apple) y CPU."""
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_description(device) -> str:
    import torch

    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def synchronize(device) -> None:
    """Espera a que terminen las operaciones asíncronas antes de medir tiempo."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy y precision/recall/F1 macro y weighted con scikit-learn."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=avg, zero_division=0
        )
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = p, r, f
    return out


def library_versions() -> dict:
    import importlib.metadata as md

    pkgs = ["torch", "transformers", "datasets", "tokenizers", "scikit-learn",
            "pandas", "numpy", "sentencepiece", "matplotlib", "psutil"]
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "no instalado"
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
