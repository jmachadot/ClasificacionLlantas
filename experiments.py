"""
Experimentos comparativos que cubren los entregables del enunciado:

  (A) Comparación de arquitecturas: CNN desde cero vs transfer learning.
  (B) Estudio de ablación sobre >= 2 factores.
  (C) Estudio de desbalance de clases y estrategias de mitigación.

Cada experimento reutiliza train.run() y agrega los resultados en CSV/JSON.
"""
import copy
import csv
import json
from pathlib import Path

import torch

import config as C
from train import run


def _base_cfg(**overrides) -> C.Config:
    cfg = C.Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _save_table(rows, out_path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[tabla] {out_path}")


def _row(name, metrics, extra=None):
    r = {"experiment": name}
    if extra:
        r.update(extra)
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        r[k] = round(float(metrics.get(k, float("nan"))), 4)
    return r


# --------------------------------------------------------------------------- #
# (A) Comparación de arquitecturas
# --------------------------------------------------------------------------- #
def compare_architectures(device, base: C.Config, out_dir: Path):
    rows = []
    configs = [
        ("custom_scratch", _base_cfg(
            model_name="custom", pretrained=False, lr=1e-3,
            run_name="A_custom_scratch", **_shared(base))),
        ("resnet50_transfer", _base_cfg(
            model_name="resnet50", pretrained=True, lr=1e-4,
            run_name="A_resnet50_transfer", **_shared(base))),
        ("efficientnet_b0_transfer", _base_cfg(
            model_name="efficientnet_b0", pretrained=True, lr=1e-4,
            run_name="A_efficientnet_transfer", **_shared(base))),
    ]
    for name, cfg in configs:
        metrics, _ = run(cfg, device)
        rows.append(_row(name, metrics, {"model": cfg.model_name}))
    _save_table(rows, out_dir / "comparison_architectures.csv")
    return rows


# --------------------------------------------------------------------------- #
# (B) Ablación
# --------------------------------------------------------------------------- #
def ablation_study(device, base: C.Config, out_dir: Path):
    """Ablación sobre 3 factores: loss, aumento de datos, profundidad del backbone."""
    rows = []
    experiments = [
        # Factor 1: función de pérdida (BCE vs Focal)
        ("loss=bce",   _base_cfg(model_name="resnet50", loss="bce",
                                  run_name="B_loss_bce", **_shared(base))),
        ("loss=focal", _base_cfg(model_name="resnet50", loss="focal",
                                  run_name="B_loss_focal", **_shared(base))),
        # Factor 2: aumento de datos
        ("aug=none",     _base_cfg(model_name="resnet50", aug_strength="none",
                                   run_name="B_aug_none", **_shared(base))),
        ("aug=standard", _base_cfg(model_name="resnet50", aug_strength="standard",
                                   run_name="B_aug_standard", **_shared(base))),
        ("aug=strong",   _base_cfg(model_name="resnet50", aug_strength="strong",
                                   run_name="B_aug_strong", **_shared(base))),
        # Factor 3: profundidad del backbone
        ("backbone=resnet18", _base_cfg(model_name="resnet18",
                                        run_name="B_resnet18", **_shared(base))),
        ("backbone=resnet50", _base_cfg(model_name="resnet50",
                                        run_name="B_resnet50", **_shared(base))),
    ]
    for name, cfg in experiments:
        metrics, _ = run(cfg, device, make_gradcam=False)
        rows.append(_row(name, metrics))
    _save_table(rows, out_dir / "ablation.csv")
    return rows


# --------------------------------------------------------------------------- #
# (C) Estudio de desbalance
# --------------------------------------------------------------------------- #
def imbalance_study(device, base: C.Config, out_dir: Path):
    rows = []
    experiments = [
        ("none",          _base_cfg(model_name="resnet50", imbalance_strategy="none",
                                    loss="bce", run_name="C_imb_none", **_shared(base))),
        ("class_weights", _base_cfg(model_name="resnet50", imbalance_strategy="class_weights",
                                    loss="bce", run_name="C_imb_classweights", **_shared(base))),
        ("oversampling",  _base_cfg(model_name="resnet50", imbalance_strategy="oversampling",
                                    loss="bce", run_name="C_imb_oversampling", **_shared(base))),
        ("focal_loss",    _base_cfg(model_name="resnet50", imbalance_strategy="none",
                                    loss="focal", run_name="C_imb_focal", **_shared(base))),
    ]
    for name, cfg in experiments:
        metrics, _ = run(cfg, device, make_gradcam=False)
        rows.append(_row(name, metrics, {"strategy": name}))
    _save_table(rows, out_dir / "imbalance_study.csv")
    return rows


def _shared(base: C.Config):
    """Hiperparámetros compartidos (datos/entrenamiento) que se propagan a cada experimento."""
    return dict(
        data_root=base.data_root, img_size=base.img_size,
        epochs=base.epochs, batch_size=base.batch_size,
        num_workers=base.num_workers, seed=base.seed,
        output_dir=base.output_dir,
    )
