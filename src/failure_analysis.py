"""
Análisis cualitativo de fallos: identifica los ejemplos peor clasificados
(mayor confianza en la clase equivocada) y los guarda en una cuadrícula con
hipótesis para su discusión en el informe.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as C


def collect_failures(y_true, y_prob, paths, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    failures = []
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            # "confianza" en la predicción errónea
            conf = y_prob[i] if y_pred[i] == 1 else (1 - y_prob[i])
            failures.append({
                "path": paths[i],
                "true": int(y_true[i]),
                "pred": int(y_pred[i]),
                "prob_damaged": float(y_prob[i]),
                "confidence": float(conf),
                "type": "FN" if (y_true[i] == 1 and y_pred[i] == 0) else
                        ("FP" if (y_true[i] == 0 and y_pred[i] == 1) else "?"),
            })
    failures.sort(key=lambda d: d["confidence"], reverse=True)
    return failures


def save_failure_grid(failures, out_path: Path, n=6):
    sel = failures[:n]
    if not sel:
        print("[failures] No hubo clasificaciones erróneas.")
        return
    cols = 3
    rows = (len(sel) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, f in zip(axes, sel):
        try:
            img = Image.open(f["path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            pass
        ax.set_title(
            f"{f['type']} | real={C.CLASS_NAMES[f['true']]} "
            f"pred={C.CLASS_NAMES[f['pred']]}\nP(daño)={f['prob_damaged']:.2f}",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
