"""
Grad-CAM (Selvaraju et al., 2017) para interpretabilidad.

Registra hooks sobre la última capa convolucional para capturar activaciones y
gradientes, y produce un mapa de calor que indica qué regiones de la textura
activan la clasificación de "daño".
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config as C


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self.handles:
            h.remove()

    def __call__(self, input_tensor):
        """input_tensor: [1, 3, H, W]. Devuelve (cam [H,W] en [0,1], prob)."""
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        logit = self.model(input_tensor).view(-1)  # logit de "damaged"
        prob = torch.sigmoid(logit).item()

        self.model.zero_grad(set_to_none=True)
        logit.backward(retain_graph=False)

        # Pesos = promedio espacial de los gradientes
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1,C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1,1,h,w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, prob


def denormalize(tensor):
    """Tensor normalizado [3,H,W] -> imagen RGB uint8."""
    mean = np.array(C.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(C.IMAGENET_STD).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * std + mean
    img = np.clip(img, 0, 1).transpose(1, 2, 0)
    return (img * 255).astype(np.uint8)


def overlay_cam(rgb_uint8, cam, alpha=0.45):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * rgb_uint8)
    return overlay


def save_gradcam_grid(model, target_layer, dataset, device, out_path: Path,
                      indices, n_max=6):
    """Genera una cuadrícula imagen original | Grad-CAM para los índices dados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam_engine = GradCAM(model, target_layer, device)
    indices = indices[:n_max]
    fig, axes = plt.subplots(len(indices), 2, figsize=(6, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        img_t, label, path = dataset[idx]
        cam, prob = cam_engine(img_t.unsqueeze(0))
        rgb = denormalize(img_t)
        overlay = overlay_cam(rgb, cam)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Real: {C.CLASS_NAMES[int(label)]}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM | P(daño)={prob:.2f}", fontsize=9)
        axes[row, 1].axis("off")

    cam_engine.remove()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
