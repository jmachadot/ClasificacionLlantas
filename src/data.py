"""
Carga de datos, particiones estratificadas, transformaciones y manejo del
desbalance de clases.

El dataset "Tire Texture Image Recognition" de Kaggle se organiza en carpetas
por clase (estilo ImageFolder). Como la nomenclatura exacta de las carpetas
puede variar, escaneamos recursivamente cualquier carpeta que contenga imágenes
y mapeamos su nombre a una etiqueta binaria mediante palabras clave
(ver config.DAMAGED_KEYWORDS / GOOD_KEYWORDS).
"""
import os
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config as C


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Escaneo del dataset
# --------------------------------------------------------------------------- #
def _infer_binary_label(folder_name: str):
    name = folder_name.lower()
    for kw in C.DAMAGED_KEYWORDS:
        if kw in name:
            return 1
    for kw in C.GOOD_KEYWORDS:
        if kw in name:
            return 0
    return None


def scan_dataset(root: str) -> List[Tuple[str, int, str]]:
    """Devuelve una lista de (ruta_imagen, etiqueta_binaria, nombre_clase_original)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de datos: {root}\n"
            f"Descarga el dataset (ver README) y colócalo en esa ruta."
        )

    samples: List[Tuple[str, int, str]] = []
    unmapped = set()

    for dirpath, _, filenames in os.walk(root):
        imgs = [f for f in filenames if Path(f).suffix.lower() in C.IMG_EXTENSIONS]
        if not imgs:
            continue
        folder = Path(dirpath).name
        label = _infer_binary_label(folder)
        if label is None:
            unmapped.add(folder)
            continue
        for fn in imgs:
            samples.append((str(Path(dirpath) / fn), label, folder))

    if not samples:
        raise RuntimeError(
            "No se pudo mapear ninguna carpeta a las clases 'good'/'damaged'.\n"
            f"Carpetas con imágenes encontradas pero no mapeadas: {sorted(unmapped)}\n"
            "Ajusta config.DAMAGED_KEYWORDS / GOOD_KEYWORDS para tu estructura."
        )
    if unmapped:
        print(f"[data] Aviso: carpetas ignoradas (sin mapeo): {sorted(unmapped)}")
    return samples


def stratified_split(samples, val_split, test_split, seed):
    """Partición estratificada train/val/test sin sklearn (para minimizar dependencias)."""
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for s in samples:
        by_label[s[1]].append(s)

    train, val, test = [], [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# --------------------------------------------------------------------------- #
# Transformaciones
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int, train: bool, aug_strength: str = "standard"):
    normalize = transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)

    if not train or aug_strength == "none":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    if aug_strength == "standard":
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ])

    # "strong": aumento orientado, útil para mitigar desbalance / sobreajuste
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TireDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32), path


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def get_dataloaders(cfg: C.Config):
    set_seed(cfg.seed)
    samples = scan_dataset(cfg.data_root)
    train_s, val_s, test_s = stratified_split(
        samples, cfg.val_split, cfg.test_split, cfg.seed
    )

    train_tf = build_transforms(cfg.img_size, train=True, aug_strength=cfg.aug_strength)
    eval_tf = build_transforms(cfg.img_size, train=False)

    train_ds = TireDataset(train_s, train_tf)
    val_ds = TireDataset(val_s, eval_tf)
    test_ds = TireDataset(test_s, eval_tf)

    # Conteo de clases en train (para desbalance)
    train_labels = [s[1] for s in train_s]
    counts = Counter(train_labels)
    n_neg = counts.get(0, 0)
    n_pos = counts.get(1, 0)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    # Estrategia de muestreo para mitigar desbalance
    sampler = None
    shuffle = True
    if cfg.imbalance_strategy == "oversampling" and n_pos > 0 and n_neg > 0:
        class_w = {0: 1.0 / n_neg, 1: 1.0 / n_pos}
        weights = [class_w[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    info = {
        "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
        "train_class_counts": {"good": n_neg, "damaged": n_pos},
        "pos_weight": pos_weight,
        "imbalance_ratio": (max(n_neg, n_pos) / max(min(n_neg, n_pos), 1)),
    }
    return train_loader, val_loader, test_loader, info
