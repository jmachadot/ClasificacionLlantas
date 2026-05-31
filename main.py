"""
Punto de entrada de extremo a extremo (un solo comando).

  python main.py                # ejecuta TODO: comparación + ablación + desbalance
  python main.py --quick        # versión rápida (pocas épocas) para verificar el pipeline
  python main.py --stage compare    # solo comparación de arquitecturas
  python main.py --stage ablation   # solo ablación
  python main.py --stage imbalance  # solo desbalance

Genera un resumen en outputs/SUMMARY.md
"""
import argparse
import json
from pathlib import Path

import torch

import config as C
import experiments as E


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(C.PROJECT_ROOT / "data"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="3 épocas, batch pequeño: smoke test del pipeline")
    p.add_argument("--stage", choices=["all", "compare", "ablation", "imbalance"],
                   default="all")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("AVISO: no se detectó GPU; el entrenamiento será lento.")

    base = C.Config(
        data_root=args.data_root,
        epochs=3 if args.quick else args.epochs,
        batch_size=16 if args.quick else args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    out_dir = Path(base.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.stage in ("all", "compare"):
        results["architectures"] = E.compare_architectures(device, base, out_dir)
    if args.stage in ("all", "ablation"):
        results["ablation"] = E.ablation_study(device, base, out_dir)
    if args.stage in ("all", "imbalance"):
        results["imbalance"] = E.imbalance_study(device, base, out_dir)

    # Resumen
    summary = ["# Resumen de resultados\n"]
    for section, rows in results.items():
        summary.append(f"\n## {section}\n")
        if rows:
            keys = [k for k in rows[0].keys()]
            summary.append("| " + " | ".join(keys) + " |")
            summary.append("|" + "|".join(["---"] * len(keys)) + "|")
            for r in rows:
                summary.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nListo. Resumen en {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
