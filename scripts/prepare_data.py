"""
Descarga y verificación del dataset.

Opción 1 (recomendada) — Kaggle API:
  1. pip install kaggle
  2. Coloca tu kaggle.json en ~/.kaggle/ (Linux/Mac) o C:\\Users\\<TU_USUARIO>\\.kaggle\\ (Windows)
  3. python scripts/prepare_data.py --download

Opción 2 — manual:
  Descarga el ZIP desde
  https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition
  y descomprímelo dentro de la carpeta data/ del proyecto.

Luego verifica con:
  python scripts/prepare_data.py --verify
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402

DATASET = "jehanbhathena/tire-texture-image-recognition"


def download(data_root: Path):
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {DATASET} en {data_root} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET,
             "-p", str(data_root), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: no se encontró el comando 'kaggle'. Instala con: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR al descargar: {e}\nRevisa que kaggle.json esté configurado.")
        sys.exit(1)
    print("Descarga completa.")


def verify(data_root: Path):
    from src.data import scan_dataset
    samples = scan_dataset(str(data_root))
    n_good = sum(1 for s in samples if s[1] == 0)
    n_dmg = sum(1 for s in samples if s[1] == 1)
    folders = sorted({s[2] for s in samples})
    print(f"Total imágenes: {len(samples)}")
    print(f"  good (0):    {n_good}")
    print(f"  damaged (1): {n_dmg}")
    print(f"Carpetas mapeadas: {folders}")
    if n_good == 0 or n_dmg == 0:
        print("AVISO: una de las clases está vacía. Revisa los nombres de carpeta "
              "y ajusta config.DAMAGED_KEYWORDS / GOOD_KEYWORDS si es necesario.")
    else:
        print("OK: ambas clases presentes.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(C.PROJECT_ROOT / "data"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    root = Path(args.data_root)
    if args.download:
        download(root)
    if args.verify or not args.download:
        verify(root)
