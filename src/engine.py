"""
Motor de entrenamiento y evaluación: bucles de época, validación, early
stopping, guardado de checkpoints e historial.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.metrics import compute_metrics, format_metrics


@torch.no_grad()
def predict(model, loader, device):
    """Devuelve (y_true, y_prob, paths)."""
    model.eval()
    ys, ps, paths = [], [], []
    for imgs, labels, batch_paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        ps.append(probs)
        ys.append(labels.numpy())
        paths.extend(batch_paths)
    return np.concatenate(ys), np.concatenate(ps), paths


def _build_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=0.9, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer, cfg):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    return None


def train_model(model, train_loader, val_loader, criterion, cfg, device,
                run_dir: Path = None, verbose=True):
    model.to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda")) # type: ignore

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, disable=not verbose,
                    desc=f"Época {epoch}/{cfg.epochs}")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")): # type: ignore
                logits = model(imgs).view(-1)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            pbar.set_postfix(loss=running_loss / max(n, 1))

        if scheduler is not None:
            scheduler.step()

        # Validación
        y_true, y_prob, _ = predict(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_prob)
        train_loss = running_loss / max(n, 1)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
        })
        if verbose:
            print(f"  [val] loss_train={train_loss:.4f} | {format_metrics(val_metrics)}")

        # Early stopping según F1 de validación
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                if verbose:
                    print(f"  Early stopping en época {epoch} (mejor época: {best_epoch}).")
                break

    model.load_state_dict(best_state)

    if run_dir is not None:
        torch.save(best_state, run_dir / "best_model.pt")
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return model, history, best_epoch
