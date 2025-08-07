# Copyright (C) 2025 Miguel Marina
# Author: Miguel Marina <karel.capek.robotics@gmail.com>
# LinkedIn: https://www.linkedin.com/in/progman32/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image


def gaussian_kernel(size: int = 5, sigma: float = 1.4) -> np.ndarray:
    ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return kernel / kernel.sum()


def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    m, n = image.shape
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for i in range(m):
        for j in range(n):
            region = padded[i : i + ky, j : j + kx]
            out[i, j] = np.sum(region * kernel)
    return out


def non_max_suppression(gradient_mag: np.ndarray, gradient_dir: np.ndarray) -> np.ndarray:
    M, N = gradient_mag.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = gradient_dir % 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                q = gradient_mag[i, j + 1]
                r = gradient_mag[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_mag[i + 1, j - 1]
                r = gradient_mag[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_mag[i + 1, j]
                r = gradient_mag[i - 1, j]
            else:
                q = gradient_mag[i - 1, j - 1]
                r = gradient_mag[i + 1, j + 1]
            if (gradient_mag[i, j] >= q) and (gradient_mag[i, j] >= r):
                Z[i, j] = gradient_mag[i, j]
    return Z


def double_threshold_and_hysteresis(img: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    strong_val = 255
    weak_val = 75
    strong_mask = img >= high_threshold
    weak_mask = (img >= low_threshold) & (img < high_threshold)
    strong = np.zeros_like(img, dtype=np.uint8)
    weak = np.zeros_like(img, dtype=np.uint8)
    strong[strong_mask] = strong_val
    weak[weak_mask] = weak_val
    out = strong.copy()
    M, N = img.shape
    changed = True
    while changed:
        changed = False
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if weak[i, j] == weak_val and out[i, j] != strong_val:
                    neighbors = out[i - 1 : i + 2, j - 1 : j + 2]
                    if (neighbors == strong_val).any():
                        out[i, j] = strong_val
                        changed = True
        if not changed:
            break
    return (out == strong_val).astype(np.uint8) * 255


def canny_edge_detection(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 150) -> np.ndarray:
    gauss = gaussian_kernel(5, 1.4)
    smooth = apply_kernel(image, gauss)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    Ix = apply_kernel(smooth, Kx)
    Iy = apply_kernel(smooth, Ky)
    grad_mag = np.sqrt(Ix ** 2 + Iy ** 2)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max() * 255.0
    else:
        grad_mag[:] = 0.0
    grad_dir = np.rad2deg(np.arctan2(Iy, Ix))
    nms = non_max_suppression(grad_mag, grad_dir)
    return double_threshold_and_hysteresis(nms, low_threshold, high_threshold)


def morph_dilate(binary_image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    m, n = binary_image.shape
    pad = kernel_size // 2
    out = binary_image.copy()
    for _ in range(iterations):
        temp = out.copy()
        for i in range(m):
            for j in range(n):
                region = out[max(i - pad, 0) : min(i + pad + 1, m), max(j - pad, 0) : min(j + pad + 1, n)]
                if np.any(region == 255):
                    temp[i, j] = 255
        out = temp
    return out


def load_process_shape_image(
    path: str,
    target_size: int = 28,
    canny_low: float = 50,
    canny_high: float = 150,
    dilation_iter: int = 1,
    angles: Iterable[int] = range(0, 181, 10),
) -> List[np.ndarray]:
    try:
        resample_bicubic = Image.Resampling.BICUBIC
        resample_lanczos = Image.Resampling.LANCZOS
    except AttributeError:
        resample_bicubic = Image.BICUBIC
        resample_lanczos = Image.LANCZOS

    pil_img = Image.open(path).convert("L")
    # Rotate around the background color to avoid introducing spurious edges
    # when the canvas is expanded. Without specifying ``fillcolor`` the newly
    # exposed corners are filled with black which the edge detector interprets
    # as strong gradients, resulting in thick square artefacts after rotation.
    bg_color = pil_img.getpixel((0, 0))
    processed_images = []
    for angle in angles:
        rotated = pil_img.rotate(
            angle, resample=resample_bicubic, expand=True, fillcolor=bg_color
        )
        arr = np.array(rotated, dtype=np.float32)
        edges = canny_edge_detection(arr, canny_low, canny_high)
        dilated = morph_dilate(edges, 3, dilation_iter)
        norm_img = np.array(
            Image.fromarray(dilated.astype(np.uint8)).resize((target_size, target_size), resample=resample_lanczos),
            dtype=np.float32,
        ) / 255.0
        processed_images.append(norm_img.flatten())
    return processed_images


def load_annotated_dataset(root_dir: str) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load a dataset with COCO or YOLO-style annotations.

    Parameters
    ----------
    root_dir:
        Path to a directory containing the dataset.  For COCO-style
        annotations the directory must include an ``annotations.json`` file.
        For YOLO-style datasets a ``images/`` directory with the images and a
        corresponding ``labels/`` directory with text files are expected.

    Returns
    -------
    list of tuples
        Each element contains ``(image_tensor, boxes, labels)`` where
        ``image_tensor`` is a ``(C, H, W)`` float tensor in ``[0, 1]``.
        ``boxes`` is an ``(N, 4)`` tensor of ``[x1, y1, x2, y2]`` in absolute
        pixel coordinates and ``labels`` is a ``(N,)`` tensor of class indices.
    """

    root = Path(root_dir)
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    anno_file = root / "annotations.json"
    if anno_file.exists():
        with open(anno_file, "r", encoding="utf-8") as fh:
            coco = json.load(fh)
        img_index = {img["id"]: img for img in coco.get("images", [])}
        anns_by_img: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_id, img_info in img_index.items():
            file_name = img_info.get("file_name", "")
            img_path = root / file_name
            if not img_path.exists():
                img_path = root / "images" / file_name
            if not img_path.exists():
                continue
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(pil_img).transpose(2, 0, 1)).float() / 255.0
            boxes_list = []
            labels_list = []
            for ann in anns_by_img.get(img_id, []):
                x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                boxes_list.append([x, y, x + w, y + h])
                labels_list.append(int(ann.get("category_id", 0)))
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
            samples.append((img_tensor, boxes, labels))
        return samples

    # YOLO-style dataset
    img_dir = root / "images"
    label_dir = root / "labels"
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_dir / (img_path.stem + ".txt")
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size
        img_tensor = torch.from_numpy(np.array(pil_img).transpose(2, 0, 1)).float() / 255.0
        boxes_list = []
        labels_list = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    x2 = (xc + bw / 2) * w
                    y2 = (yc + bh / 2) * h
                    boxes_list.append([x1, y1, x2, y2])
                    labels_list.append(int(cls))
        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.int64)
        samples.append((img_tensor, boxes, labels))
    return samples
