# Cómo compilar el informe (formato NeurIPS)

> **¿Prefieres Word?** En esta misma carpeta tienes `Informe_Llantas_Plantilla.docx`,
> con la misma estructura, tablas y huecos. Ábrelo en Word, rellena lo marcado en
> rojo, e inserta las figuras (`Insertar > Imágenes`) en los recuadros grises.
> El resto de este documento es solo para la versión LaTeX.


El informe está en `main.tex` y usa el estilo oficial de NeurIPS. Tienes dos caminos.

## Opción 1 — Overleaf (la más simple, recomendada)

1. Entra a https://www.overleaf.com y crea un proyecto desde la plantilla
   **"NeurIPS 2024"** (busca "NeurIPS" en *Templates*). Esto ya trae
   `neurips_2024.sty`.
2. Reemplaza el `main.tex` de la plantilla por el de esta carpeta.
3. Sube también `references.bib` y, dentro de una carpeta `figures/`, las imágenes
   que generó el código (ver abajo).
4. Compila (Overleaf ejecuta `pdflatex` + `bibtex` automáticamente).

## Opción 2 — Local (TeX Live / MiKTeX)

1. Descarga el archivo de estilo `neurips_2024.sty` desde el sitio oficial de NeurIPS
   (página "Paper Submission Instructions" / "Style Files") y colócalo en esta carpeta,
   junto a `main.tex`.
2. Compila:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```

> Si no consigues `neurips_2024.sty`, abre `main.tex`, comenta la línea
> `\usepackage[final]{neurips_2024}` y descomenta el bloque **FALLBACK** (usa la clase
> `article` con márgenes y tipografía aproximados). El contenido compilará igual; solo
> cambia el formato exacto.

## Copiar las figuras generadas por el código

El código deja las figuras en `outputs/<run_name>/`. Cópialas a `informe/figures/`
con estos nombres (los que espera `main.tex`):

| Origen (ejemplo)                                  | Destino en `figures/`  |
|---------------------------------------------------|------------------------|
| `outputs/resnet50_main/confusion_matrix.png`      | `confusion_matrix.png` |
| `outputs/resnet50_main/roc_curve.png`             | `roc_curve.png`        |
| `outputs/resnet50_main/gradcam.png`               | `gradcam.png`          |
| `outputs/resnet50_main/failures.png`              | `failures.png`         |

Las tablas (`comparison_architectures.csv`, `ablation.csv`, `imbalance_study.csv`) se
copian a mano en las tablas de `main.tex` donde dice `--`.

## Antes de entregar

- Rellena todos los `\TODO{...}` (aparecen en rojo en el PDF).
- Pon los nombres del grupo en `\author{...}`.
- Verifica que el informe tenga 6–10 páginas y ≥5 referencias en `references.bib`.
