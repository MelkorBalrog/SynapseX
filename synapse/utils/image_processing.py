"""Image processing utilities for SynapseX."""
import os
import math
import numpy as np
from PIL import Image
from numba import njit


def gaussian_kernel(size=5, sigma=1.4):
    ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return kernel / kernel.sum()


@njit
def apply_kernel_numba(image, kernel):
    m, n = image.shape
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    padded = np.empty((m + 2 * pad_y, n + 2 * pad_x), dtype=np.float32)
    for i in range(m + 2 * pad_y):
        for j in range(n + 2 * pad_x):
            ii = i - pad_y
            jj = j - pad_x
            if ii < 0:
                ii = -ii
            elif ii >= m:
                ii = 2 * m - ii - 2
            if jj < 0:
                jj = -jj
            elif jj >= n:
                jj = 2 * n - jj - 2
            padded[i, j] = image[ii, jj]
    out = np.empty((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for a in range(ky):
                for b in range(kx):
                    s += padded[i + a, j + b] * kernel[a, b]
            out[i, j] = s
    return out


def apply_kernel(image, kernel):
    return apply_kernel_numba(image, kernel)


# Additional processing functions (canny_edge_detection, morph_dilate, etc.) could be
# added here as needed. The original project contained many more utilities; they were
# trimmed for brevity in this refactor.
