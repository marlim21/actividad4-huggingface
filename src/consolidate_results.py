"""Une las tablas de cada ejecución en results/model_comparison.csv.

Busca results/<ejecucion>/model_comparison.csv (p. ej. colab_gpu, local_mps,
local_cpu) y agrega la columna `ejecucion` para distinguir el entorno.

Uso:
  python src/consolidate_results.py
"""
from __future__ import annotations

import pandas as pd

import utils

ORDER = ["colab_gpu", "local_mps", "local_cpu", "notebook_local"]


def main() -> None:
    frames = []
    runs = sorted(p.parent.name for p in utils.RESULTS_DIR.glob("*/model_comparison.csv"))
    runs = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]
    for run in runs:
        df = pd.read_csv(utils.RESULTS_DIR / run / "model_comparison.csv")
        df.insert(0, "ejecucion", run)
        frames.append(df)
    if not frames:
        raise SystemExit("No hay resultados. Ejecute antes src/evaluate_models.py")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(utils.RESULTS_DIR / "model_comparison.csv", index=False)
    print(out[["ejecucion", "modelo", "f1_macro", "accuracy", "rendimiento_reg_s", "online_p95_ms", "dispositivo"]]
          .to_string(index=False))
    print(f"\nEjecuciones incluidas: {runs}")


if __name__ == "__main__":
    main()
