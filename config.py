"""
Configuración central del proyecto.

Todos los hiperparámetros y rutas se definen aquí. Los scripts de
entrenamiento/evaluación aceptan argumentos de línea de comandos que
sobreescriben estos valores por defecto.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent

# Estadísticas de normalización de ImageNet (necesarias para backbones preentrenados)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Extensiones de imagen reconocidas al escanear el dataset
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Palabras clave para mapear los nombres de carpeta del dataset a etiquetas binarias.
# Clase positiva (label = 1) = LLANTA DAÑADA  (lo que nos interesa detectar)
# Clase negativa (label = 0) = LLANTA EN BUEN ESTADO
DAMAGED_KEYWORDS = ("crack", "defect", "damage", "damaged", "bad", "worn", "flat", "broken", "danad")
GOOD_KEYWORDS = ("good", "normal", "fine", "intact", "new", "healthy", "buen")

CLASS_NAMES = ("good", "damaged")  # índice 0 -> good, índice 1 -> damaged


@dataclass
class Config:
    # --- Datos ---
    data_root: str = str(PROJECT_ROOT / "data")
    img_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    seed: int = 42

    # --- Modelo ---
    # opciones: "custom", "resnet18", "resnet50", "efficientnet_b0"
    model_name: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.5

    # --- Entrenamiento ---
    epochs: int = 25
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"          # "adamw" | "sgd"
    scheduler: str = "cosine"         # "cosine" | "step" | "none"
    early_stopping_patience: int = 7
    label_smoothing: float = 0.0

    # --- Pérdida ---
    loss: str = "bce"                 # "bce" | "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # --- Manejo de desbalance de clases ---
    # "none" | "class_weights" | "oversampling"
    imbalance_strategy: str = "class_weights"

    # --- Aumento de datos ---
    aug_strength: str = "standard"    # "none" | "standard" | "strong"

    # --- Salidas ---
    output_dir: str = str(PROJECT_ROOT / "outputs")
    run_name: str = "default"

    def run_dir(self) -> Path:
        d = Path(self.output_dir) / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
