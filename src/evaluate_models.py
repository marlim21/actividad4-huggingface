"""Evalúa los tres modelos preentrenados sobre el mismo dataset y en las mismas condiciones.

Para cada modelo: carga (cronometrada) -> calentamiento -> inferencia en lotes
repetida (cronometrada) -> inferencia en línea con batch_size=1 -> métricas.

Requiere haber ejecutado antes:  python src/prepare_data.py

Salidas en --out (por defecto results/):
  model_comparison.csv     una fila por modelo (tabla comparativa principal)
  per_class_metrics.csv    precision/recall/F1 por clase y modelo
  predictions.csv          id, etiqueta real y predicción de cada modelo (sin texto)
  run_metadata.json        entorno, versiones, configuración y umbrales
  figures/                 matrices de confusión y gráfica calidad vs. rendimiento

Uso:
  python src/evaluate_models.py                      # dispositivo automático
  python src/evaluate_models.py --device cpu --out results/cpu
  python src/evaluate_models.py --limit 300          # prueba rápida
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import utils


def weights_size_mb(model_id: str, revision: str) -> float | None:
    """Tamaño del archivo de pesos en el Hub (safetensors si existe, si no .bin)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
        sizes = {s.rfilename: s.size for s in info.siblings}
        for name in ("model.safetensors", "pytorch_model.bin"):
            if sizes.get(name):
                return round(sizes[name] / 1024**2, 1)
    except Exception as exc:  # sin red o API no disponible
        print(f"  No se pudo consultar el tamaño en el Hub: {exc}")
    return None


def memory_snapshot(device) -> float:
    """Memoria en MB atribuible al proceso/dispositivo."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1024**2
    return psutil.Process().memory_info().rss / 1024**2


@torch.inference_mode()
def predict(model, tokenizer, texts, device, batch_size, max_length):
    preds = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        preds.append(model(**enc).logits.argmax(dim=-1).cpu())
    utils.synchronize(device)
    return torch.cat(preds).numpy()


def evaluate_model(model_id, revision, df, device, cfg):
    name = utils.SHORT_NAMES[model_id]
    print(f"\n=== {name}: {model_id}@{revision[:7]}")
    texts = df["text"].tolist()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    mem_before = memory_snapshot(device)

    # Carga (se separa del tiempo de inferencia)
    with utils.Timer() as t_load:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        model.to(device).eval()
        utils.synchronize(device)
    n_params = sum(p.numel() for p in model.parameters())
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    id2target = {k: utils.normalize_model_label(v) for k, v in id2label.items()}
    print(f"  Carga: {t_load.seconds:.1f}s | parámetros: {n_params/1e6:.1f} M | etiquetas: {id2label}")

    # Calentamiento
    predict(model, tokenizer, texts[: cfg["batch_size"] * cfg["warmup_batches"]],
            device, cfg["batch_size"], cfg["max_length"])

    # Inferencia en lote, repetida
    times, raw = [], None
    for r in range(cfg["timed_repeats"]):
        with utils.Timer() as t:
            out = predict(model, tokenizer, texts, device, cfg["batch_size"], cfg["max_length"])
        times.append(t.seconds)
        if raw is None:
            raw = out
        elif not np.array_equal(raw, out):
            print("  Aviso: predicciones distintas entre repeticiones")
        print(f"  Pasada {r + 1}: {t.seconds:.2f}s")
    mem_peak = memory_snapshot(device)

    # Inferencia en línea (una reseña a la vez)
    online_texts = df.sample(n=min(cfg["online_samples"], len(df)), random_state=utils.SEED)["text"].tolist()
    online_ms = []
    for txt in online_texts:
        with utils.Timer() as t:
            predict(model, tokenizer, [txt], device, 1, cfg["max_length"])
        online_ms.append(t.seconds * 1000)

    y_pred = [id2target[i] for i in raw]
    y_true = df["sentiment"].tolist()
    metrics = utils.compute_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=utils.LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=utils.LABELS)

    extra = {}
    if model_id.startswith("nlptown/"):  # métrica secundaria en la escala original de 5 estrellas
        stars_pred = raw + 1
        extra["stars_exact_acc"] = float((stars_pred == df["stars"].values).mean())
        extra["stars_off_by_1_acc"] = float((np.abs(stars_pred - df["stars"].values) <= 1).mean())

    total = float(np.mean(times))
    row = {
        "modelo": name,
        "model_id": model_id,
        "revision": revision,
        **{k: round(v, 4) for k, v in metrics.items()},
        "recall_negativo": round(report["negativo"]["recall"], 4),
        "f1_neutral": round(report["neutral"]["f1-score"], 4),
        "n_registros": len(df),
        "tiempo_total_s": round(total, 3),
        "tiempo_total_std_s": round(float(np.std(times)), 3),
        "latencia_media_ms": round(total / len(df) * 1000, 3),
        "rendimiento_reg_s": round(len(df) / total, 1),
        "online_media_ms": round(float(np.mean(online_ms)), 2),
        "online_p50_ms": round(float(np.percentile(online_ms, 50)), 2),
        "online_p95_ms": round(float(np.percentile(online_ms, 95)), 2),
        "tiempo_carga_s": round(t_load.seconds, 2),
        "parametros_M": round(n_params / 1e6, 1),
        "pesos_hub_MB": weights_size_mb(model_id, revision),
        "memoria_pico_MB": round(mem_peak, 0),
        "memoria_incremento_MB": round(mem_peak - mem_before, 0),
        "dispositivo": utils.device_description(device),
        "batch_size": cfg["batch_size"],
        "max_length": cfg["max_length"],
        **{k: round(v, 4) for k, v in extra.items()},
    }
    per_class = [
        {"modelo": name, "clase": c, "precision": report[c]["precision"], "recall": report[c]["recall"],
         "f1": report[c]["f1-score"], "soporte": int(report[c]["support"])}
        for c in utils.LABELS
    ]
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return row, per_class, cm, y_pred


def save_figures(rows, cms, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.4))
    for ax, row, cm in zip(np.atleast_1d(axes), rows, cms):
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(3), utils.LABELS)
        ax.set_yticks(range(3), utils.LABELS)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{row['modelo']}\nF1 macro = {row['f1_macro']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for row in rows:
        ax.scatter(row["rendimiento_reg_s"], row["f1_macro"], s=row["parametros_M"] * 3, alpha=0.7)
        ax.annotate(f"{row['modelo']}\n{row['parametros_M']:.0f} M parám.",
                    (row["rendimiento_reg_s"], row["f1_macro"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)
    ax.axhline(utils.ACCEPTANCE["f1_macro_min"], ls="--", lw=1, color="gray")
    ax.set_xlabel("Rendimiento (reseñas/s, lote de 32)")
    ax.set_ylabel("F1 macro")
    ax.set_title(f"Calidad vs. eficiencia — {rows[0]['dispositivo']}", fontsize=10)
    ax.margins(x=0.35, y=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "f1_vs_throughput.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--out", default=str(utils.RESULTS_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Evaluar solo N registros (pruebas rápidas)")
    parser.add_argument("--batch-size", type=int, default=utils.INFERENCE["batch_size"])
    parser.add_argument("--repeats", type=int, default=utils.INFERENCE["timed_repeats"])
    args = parser.parse_args()

    from pathlib import Path

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.set_seed()
    cfg = dict(utils.INFERENCE, batch_size=args.batch_size, timed_repeats=args.repeats)

    data_path = utils.DATA_DIR / "dataset.csv"
    if not data_path.exists():
        raise SystemExit("No existe data/processed/dataset.csv. Ejecute antes: python src/prepare_data.py")
    df = pd.read_csv(data_path, keep_default_na=False)
    if args.limit:
        df = df.sample(n=args.limit, random_state=utils.SEED).reset_index(drop=True)

    device = utils.get_device(args.device)
    print(f"Dispositivo: {device} — {utils.device_description(device)} | registros: {len(df)}")
    if device.type != "cuda":
        print("AVISO: no se está usando GPU CUDA; los tiempos no son comparables con Colab GPU.")

    rows, per_class, cms = [], [], []
    preds = df[["review_id", "stars", "sentiment"]].copy()
    for model_id, rev in utils.MODELS.items():
        row, pc, cm, y_pred = evaluate_model(model_id, rev, df, device, cfg)
        rows.append(row)
        per_class.extend(pc)
        cms.append(cm)
        preds[f"pred_{row['modelo']}"] = y_pred

    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "model_comparison.csv", index=False)
    pd.DataFrame(per_class).round(4).to_csv(out_dir / "per_class_metrics.csv", index=False)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    for row, cm in zip(rows, cms):
        pd.DataFrame(cm, index=utils.LABELS, columns=utils.LABELS).to_csv(
            out_dir / f"confusion_{row['modelo']}.csv")
    save_figures(rows, cms, out_dir)

    utils.save_json({
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "device_description": utils.device_description(device),
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "versions": utils.library_versions(),
        "dataset": {"id": utils.DATASET_ID, "revision": utils.DATASET_REVISION,
                    "split": utils.DATASET_SPLIT, "n_records": len(df), "limit": args.limit},
        "models": utils.MODELS,
        "inference": cfg,
        "seed": utils.SEED,
        "acceptance_thresholds_proposed": utils.ACCEPTANCE,
    }, out_dir / "run_metadata.json")

    cols = ["modelo", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
            "latencia_media_ms", "rendimiento_reg_s", "online_p95_ms", "parametros_M", "memoria_pico_MB"]
    print("\n" + comp[cols].to_string(index=False))
    print(f"\nResultados guardados en {out_dir}")


if __name__ == "__main__":
    main()
