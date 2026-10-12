"""Prepara el dataset de evaluación a partir de la fuente oficial en Hugging Face.

Pasos: carga -> inspección -> faltantes -> duplicados -> distribución de clases
-> transformación de etiquetas (5 estrellas -> 3 sentimientos) -> verificación
-> perfil de longitudes en tokens -> escritura de archivos.

Salidas (en data/processed/):
  dataset.csv           reseñas + etiquetas (uso local; NO se publica, ver data/README.md)
  dataset_manifest.csv  id, estrellas y etiqueta, sin texto (sí se publica)
  data_profile.json     parámetros y estadísticas del proceso

Uso:
  python src/prepare_data.py
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
from datasets import load_dataset

import utils


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-token-profile", action="store_true",
                        help="No descargar tokenizadores para medir longitudes.")
    args = parser.parse_args()
    utils.set_seed()

    # 1. Carga reproducible (commit fijado)
    ds = load_dataset(utils.DATASET_ID, split=utils.DATASET_SPLIT, revision=utils.DATASET_REVISION)
    df = ds.to_pandas()
    print(f"Cargado {utils.DATASET_ID}@{utils.DATASET_REVISION[:7]} [{utils.DATASET_SPLIT}]: {df.shape}")

    # 2. Estructura
    expected = {"id", "text", "label", "label_text"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas: {missing_cols}")
    print(df.dtypes.to_string())

    # 3. Valores faltantes y textos vacíos
    nulls = df.isna().sum().to_dict()
    empty_text = int((df["text"].str.strip() == "").sum())
    if any(nulls.values()) or empty_text:
        raise ValueError(f"Hay faltantes {nulls} o textos vacíos {empty_text}")

    # 4. Duplicados (se conservan: forman parte de la partición oficial)
    dup_ids = int(df["id"].duplicated().sum())
    dup_mask = df["text"].duplicated(keep=False)
    dup_text_rows = int(df["text"].duplicated().sum())
    conflicting = (
        df[dup_mask].groupby("text")["label"].nunique().gt(1).sum() if dup_mask.any() else 0
    )
    if dup_ids:
        raise ValueError("Hay identificadores repetidos")

    # 5-7. Transformación y verificación de etiquetas
    df["stars"] = df["label"].astype(int) + 1  # label 0..4 -> 1..5 estrellas
    if not df["stars"].between(1, 5).all():
        raise ValueError("Etiquetas fuera de rango")
    if not (df["label_text"].astype(int) == df["label"]).all():
        raise ValueError("label y label_text no coinciden")
    df["sentiment"] = df["stars"].map(utils.stars_to_label)
    df["sentiment_id"] = df["sentiment"].map(utils.LABEL2ID)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()
    df = df.rename(columns={"id": "review_id"})[
        ["review_id", "text", "stars", "sentiment", "sentiment_id", "n_chars", "n_words"]
    ]

    # Perfil de longitudes en tokens (para justificar max_length)
    token_profile = {}
    if not args.skip_token_profile:
        from transformers import AutoTokenizer

        texts = df["text"].tolist()
        for model_id, rev in utils.MODELS.items():
            tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
            lengths = pd.Series([len(x) for x in tok(texts, truncation=False)["input_ids"]])
            token_profile[model_id] = {
                "mean": round(float(lengths.mean()), 1),
                "p95": int(lengths.quantile(0.95)),
                "max": int(lengths.max()),
                "pct_over_max_length": round(float((lengths > utils.INFERENCE["max_length"]).mean() * 100), 3),
            }
            print(f"Tokens {model_id}: {token_profile[model_id]}")

    # 10. Escritura
    utils.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(utils.DATA_DIR / "dataset.csv", index=False, encoding="utf-8")
    df[["review_id", "stars", "sentiment"]].to_csv(utils.DATA_DIR / "dataset_manifest.csv", index=False)

    profile = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_id": utils.DATASET_ID,
        "dataset_revision": utils.DATASET_REVISION,
        "split": utils.DATASET_SPLIT,
        "seed": utils.SEED,
        "n_records": len(df),
        "null_values": nulls,
        "empty_texts": empty_text,
        "duplicate_ids": dup_ids,
        "duplicate_text_rows": dup_text_rows,
        "duplicate_texts_with_conflicting_labels": int(conflicting),
        "stars_distribution": df["stars"].value_counts().sort_index().to_dict(),
        "sentiment_distribution": df["sentiment"].value_counts().reindex(utils.LABELS).to_dict(),
        "chars": df["n_chars"].describe().round(1).to_dict(),
        "words": df["n_words"].describe().round(1).to_dict(),
        "max_length": utils.INFERENCE["max_length"],
        "token_profile": token_profile,
    }
    utils.save_json(profile, utils.DATA_DIR / "data_profile.json")
    print(f"Distribución de sentimiento: {profile['sentiment_distribution']}")
    print(f"Duplicados de texto: {dup_text_rows} (con etiqueta en conflicto: {conflicting})")
    print(f"Archivos escritos en {utils.DATA_DIR}")


if __name__ == "__main__":
    main()
