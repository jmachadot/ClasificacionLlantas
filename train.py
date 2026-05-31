"""
Entrena y evalúa UN modelo de forma completa, generando todos los artefactos:
métricas en test, matriz de confusión, curva ROC, mapas Grad-CAM y análisis de
fallos.

Uso:
    python train.py --model resnet50 --run-name resnet50_cw --epochs 25
    python train.py --model custom   --run-name custom_baseline
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from src.data import get_dataloaders, TireDataset, build_transforms, scan_dataset, stratified_split
from src.models import build_model, get_gradcam_target_layer
from src.losses import build_loss
from src.engine import train_model, predict
from src.metrics import compute_metrics, plot_confusion_matrix, plot_roc, format_metrics
from src.gradcam import save_gradcam_grid
from src.failure_analysis import collect_failures, save_failure_grid


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento de clasificador de llantas dañadas")
    p.add_argument("--data-root", default=None)
    p.add_argument("--model", dest="model_name", default="resnet50",
                   choices=["custom", "resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--loss", choices=["bce", "focal"], default=None)
    p.add_argument("--imbalance", dest="imbalance_strategy",
                   choices=["none", "class_weights", "oversampling"], default=None)
    p.add_argument("--aug", dest="aug_strength",
                   choices=["none", "standard", "strong"], default=None)
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def make_config(args) -> C.Config:
    cfg = C.Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    if args.run_name is None:
        cfg.run_name = f"{cfg.model_name}_{cfg.loss}_{cfg.imbalance_strategy}"
    return cfg


def run(cfg: C.Config, device, make_gradcam=True, make_failures=True):
    run_dir = cfg.run_dir()
    cfg.save(run_dir / "config.json")
    print(f"\n=== RUN: {cfg.run_name} | modelo={cfg.model_name} | dispositivo={device} ===")

    train_loader, val_loader, test_loader, info = get_dataloaders(cfg)
    print(f"[data] {info}")
    with open(run_dir / "data_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    model = build_model(cfg.model_name, pretrained=cfg.pretrained, dropout=cfg.dropout)
    criterion = build_loss(cfg, pos_weight_value=info["pos_weight"], device=device)

    model, history, best_epoch = train_model(
        model, train_loader, val_loader, criterion, cfg, device, run_dir=run_dir
    )

    # --- Evaluación en TEST ---
    y_true, y_prob, paths = predict(model, test_loader, device)
    test_metrics = compute_metrics(y_true, y_prob)
    print(f"[TEST] {format_metrics(test_metrics)}")
    print(f"[TEST] matriz de confusión [[TN,FP],[FN,TP]] = {test_metrics['confusion_matrix']}")

    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    plot_confusion_matrix(test_metrics["confusion_matrix"],
                          run_dir / "confusion_matrix.png",
                          title=f"Matriz de confusión — {cfg.run_name}")
    plot_roc(y_true, y_prob, run_dir / "roc_curve.png",
             title=f"ROC — {cfg.run_name}")

    # --- Grad-CAM ---
    if make_gradcam:
        try:
            # reconstruimos el test dataset para acceder por índice
            samples = scan_dataset(cfg.data_root)
            _, _, test_s = stratified_split(samples, cfg.val_split, cfg.test_split, cfg.seed)
            eval_tf = build_transforms(cfg.img_size, train=False)
            test_ds = TireDataset(test_s, eval_tf)
            labels = np.array([s[1] for s in test_s])
            good_idx = list(np.where(labels == 0)[0][:3])
            dmg_idx = list(np.where(labels == 1)[0][:3])
            target_layer = get_gradcam_target_layer(model, cfg.model_name)
            save_gradcam_grid(model, target_layer, test_ds, device,
                              run_dir / "gradcam.png", good_idx + dmg_idx, n_max=6)
            print(f"[gradcam] guardado en {run_dir / 'gradcam.png'}")
        except Exception as e:
            print(f"[gradcam] omitido por error: {e}")

    # --- Análisis de fallos ---
    if make_failures:
        failures = collect_failures(y_true, y_prob, paths)
        with open(run_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures[:20], f, indent=2)
        save_failure_grid(failures, run_dir / "failures.png", n=6)
        print(f"[failures] {len(failures)} mal clasificados | top guardados en failures.png")

    return test_metrics, history


if __name__ == "__main__":
    args = parse_args()
    cfg = make_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(cfg, device)
