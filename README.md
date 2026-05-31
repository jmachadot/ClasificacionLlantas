# Clasificación de Llantas Dañadas mediante Reconocimiento de Textura (CNN)

Solución a la **Pregunta 1** del Examen Parcial de Redes Neuronales y Aprendizaje
Profundo. Clasificación binaria de imágenes de textura de llantas —
**buen estado (`good`)** vs **dañada (`damaged`)** — con CNN, comparando una red
entrenada desde cero contra *transfer learning*, con interpretabilidad Grad-CAM,
estudio de ablación y manejo del desbalance de clases.

El código se ejecuta de extremo a extremo con un único comando (`python main.py`).

---

> Examen Parcial - Maestría en Inteligencia Artificial · Curso de Redes Neuronales y Aprendizaje Profundo · Sección A · Grupo 7

Integrantes:
- Julio Machado Torres.
- Brigitte Scarlett Del Río Ricce.

Docente:
- Ph.D. Aldo Camargo Fernández Baca.

---

## Tabla de contenido

- [1. Requisitos](#1-requisitos)
- [2. Estructura del proyecto](#2-estructura-del-proyecto)
- [3. Configuración del entorno virtual en VS Code](#3-configuración-del-entorno-virtual-en-vs-code)
- [4. Descargar el dataset](#4-descargar-el-dataset)
- [5. Ejecución](#5-ejecución)
- [6. Salidas generadas](#6-salidas-generadas)
- [7. Cómo cada entregable del enunciado queda cubierto](#7-cómo-cada-entregable-del-enunciado-queda-cubierto)
- [8. Solución de problemas](#8-solución-de-problemas)

---

## 1. Requisitos

- Python 3.11
- GPU NVIDIA con drivers CUDA (recomendado). Funciona también en CPU, pero lento.
- VS Code con la extensión **Python** (de Microsoft).
- Cuenta de Kaggle (para descargar el dataset).

Comprueba tu GPU/driver:
```bash
nvidia-smi
```

---

## 2. Estructura del proyecto

```
tire-damage-classification/
├── main.py                  # Orquestador: ejecuta todos los experimentos (1 comando)
├── train.py                 # Entrena/evalúa UN modelo (métricas + ROC + Grad-CAM + fallos)
├── experiments.py           # Comparación de arquitecturas, ablación, desbalance
├── config.py                # Configuración central / hiperparámetros
├── requirements.txt
├── environment.yml          # Alternativa con conda
├── scripts/
│   └── prepare_data.py      # Descarga/verificación del dataset
├── src/
│   ├── data.py              # Dataset, particiones estratificadas, transforms, desbalance
│   ├── models.py            # CustomCNN + ResNet/EfficientNet (transfer learning)
│   ├── losses.py            # BCE y pérdida focal
│   ├── metrics.py           # Métricas + matriz de confusión + ROC
│   ├── engine.py            # Bucles de entrenamiento/validación, early stopping
│   ├── gradcam.py           # Grad-CAM
│   └── failure_analysis.py  # Análisis cualitativo de fallos
├── .vscode/                 # Configs de ejecución/depuración listas para usar
├── data/                    # (lo creas tú) imágenes del dataset
└── outputs/                 # (se genera) modelos, métricas, figuras, resumen
```

---

## 3. Configuración del entorno virtual en VS Code

### Opción A — `venv` (recomendada, sencilla)

Abre la carpeta del proyecto en VS Code (`Archivo → Abrir carpeta`) y, en la
terminal integrada (``Ctrl+` ``):

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Luego, **muy importante**, instala PyTorch con soporte CUDA. Elige según tu CUDA
(ver `nvidia-smi`). Para CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Para CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Sin GPU (solo CPU):
```bash
pip install torch torchvision
```
> El comando actualizado para tu sistema está en https://pytorch.org/get-started/locally/

Instala el resto:
```bash
pip install -r requirements.txt
```

### Opción B — conda
```bash
conda env create -f environment.yml
conda activate tire-cnn
```

### Seleccionar el intérprete en VS Code
`Ctrl+Shift+P` → **Python: Select Interpreter** → elige el de `.venv`
(o `tire-cnn`). El archivo `.vscode/settings.json` ya apunta a `.venv` por defecto.

Verifica que la GPU es visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Descargar el dataset

Dataset: **Tire Texture Image Recognition**
(https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition)

### Vía API de Kaggle (recomendada)
1. En Kaggle: *Account → API → Create New Token* (descarga `kaggle.json`).
2. Colócalo en:
   - Linux/macOS: `~/.kaggle/kaggle.json` (y `chmod 600 ~/.kaggle/kaggle.json`)
   - Windows: `C:\Users\<TU_USUARIO>\.kaggle\kaggle.json`
3. Descarga y verifica:
```bash
python scripts/prepare_data.py --download
```

### Manual
Descarga el ZIP desde la página de Kaggle, descomprímelo dentro de `data/`, y verifica:
```bash
python scripts/prepare_data.py --verify
```

El verificador escanea recursivamente `data/`, mapea las carpetas a las clases
`good`/`damaged` por palabras clave y reporta cuántas imágenes hay por clase.
Si los nombres de carpeta no se mapean, ajusta `DAMAGED_KEYWORDS` /
`GOOD_KEYWORDS` en `config.py`.

---

## 5. Ejecución

### Prueba rápida (smoke test, ~3 épocas)
Confirma que todo el pipeline corre antes de entrenar en serio:
```bash
python main.py --quick
```

### Pipeline completo (todos los entregables) — un solo comando
```bash
python main.py
```
Esto ejecuta y agrega:
- **Comparación de arquitecturas** (CNN desde cero vs ResNet-50 vs EfficientNet-B0)
- **Ablación** (pérdida BCE/focal, aumento none/standard/strong, profundidad ResNet18/50)
- **Estudio de desbalance** (sin mitigación, pesos de clase, sobremuestreo, focal)

Resultados agregados en `outputs/SUMMARY.md` y `outputs/all_results.json`.

### Entrenar un único modelo
```bash
# Transfer learning (ResNet-50, pesos ImageNet)
python train.py --model resnet50 --run-name resnet50_main --epochs 25

# CNN desde cero
python train.py --model custom --no-pretrained --lr 0.001 --run-name custom_main --epochs 30

# Variar pérdida / estrategia de desbalance / aumento
python train.py --model resnet50 --loss focal --imbalance oversampling --aug strong
```

### Por etapas
```bash
python main.py --stage compare
python main.py --stage ablation
python main.py --stage imbalance
```

### Desde VS Code (sin escribir comandos)
Pestaña **Run and Debug** (`Ctrl+Shift+D`) → elige una configuración:
*Verificar datos*, *Smoke test (quick)*, *Entrenar ResNet50 (transfer)*,
*Entrenar CNN desde cero* o *Pipeline completo (todo)* → ▶.

---

## 6. Salidas generadas

Por cada `run` se crea `outputs/<run_name>/` con:

| Archivo | Contenido |
|---|---|
| `config.json` | Configuración exacta usada (reproducibilidad) |
| `data_info.json` | Tamaños de partición, conteo por clase, `pos_weight`, ratio de desbalance |
| `best_model.pt` | Pesos del mejor modelo (por F1 de validación) |
| `history.json` | Curva de entrenamiento por época |
| `test_metrics.json` | Exactitud, precisión, recall, F1, AUC-ROC + matriz de confusión |
| `confusion_matrix.png` | Matriz de confusión |
| `roc_curve.png` | Curva ROC con AUC |
| `gradcam.png` | Mapas Grad-CAM para muestras `good` y `damaged` |
| `failures.png` / `failures.json` | Ejemplos peor clasificados (FN/FP) para discusión |

---

## 7. Cómo cada entregable del enunciado queda cubierto

1. **Dos arquitecturas (desde cero + transfer learning)** → `experiments.compare_architectures`
   entrena `CustomCNN`, `ResNet-50` y `EfficientNet-B0`.
2. **Evaluación comparativa** (exactitud, precisión, recall, F1, AUC-ROC + matriz de
   confusión) → `src/metrics.py`, generada para cada run.
3. **Ablación sobre ≥2 factores** → `experiments.ablation_study`: pérdida (BCE vs
   focal), aumento de datos (none/standard/strong) y profundidad del backbone
   (ResNet-18 vs ResNet-50).
4. **Interpretabilidad Grad-CAM** → `src/gradcam.py`, `gradcam.png` con mapas sobre
   ambas clases.
5. **Desbalance de clases** → `experiments.imbalance_study` compara *sin mitigación*
   contra *pesos de clase*, *sobremuestreo* y *pérdida focal*; `data_info.json`
   cuantifica el desbalance.
6. **Análisis de fallos (≥5 ejemplos)** → `src/failure_analysis.py`, ordena los
   errores por confianza y guarda los peores con su tipo (FN/FP).

---

## 8. Solución de problemas

- **`torch.cuda.is_available()` devuelve `False`** → reinstala torch con el índice
  CUDA correcto (sección 3); revisa que el driver NVIDIA esté actualizado.
- **`CUDA out of memory`** → baja `--batch-size` (p. ej. 16 u 8).
- **`No se pudo mapear ninguna carpeta`** → revisa los nombres de carpeta dentro de
  `data/` y ajusta las palabras clave en `config.py`.
- **`num_workers` lento o cuelga en Windows** → usa `--num-workers 0`.

---
