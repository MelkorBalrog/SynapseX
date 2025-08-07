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

"""Image processing utilities for SynapseX."""
import os
import math
from typing import Callable, Generator, Iterable, Tuple, Any

import numpy as np
from PIL import Image
from numba import njit


def load_and_preprocess(path: str, target_size: int) -> np.ndarray:
    """Load an image file, convert to grayscale, resize and flatten to [0,1]."""
    img = Image.open(path).convert("L").resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()

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


def stream_frames(video_source: Any) -> Generator[np.ndarray, None, None]:
    """Yield frames from a video file or camera.

    Parameters
    ----------
    video_source:
        Path to a video file or an integer camera index understood by OpenCV.

    Yields
    ------
    numpy.ndarray
        Consecutive frames read from the source in BGR format.
    """
    try:
        import cv2  # imported lazily to keep dependency optional
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise ImportError("stream_frames requires OpenCV (cv2) to be installed") from exc

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {video_source}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def stream_tracks(
    video_source: Any,
    detect_objects: Callable[[np.ndarray], Iterable[Tuple[int, int, int, int]]],
    tracking,
) -> Generator[Tuple[np.ndarray, Any], None, None]:
    """Yield frames alongside tracking results.

    For each frame obtained from :func:`stream_frames`, ``detect_objects`` is
    invoked to produce detections which are then fed into ``tracking.update``.
    The resulting track data is yielded together with the frame so callers can
    render or otherwise process the output.

    Parameters
    ----------
    video_source:
        Path or index of the video source understood by OpenCV.
    detect_objects:
        Callable returning an iterable of bounding boxes (x, y, w, h) for a
        given frame.
    tracking:
        Object exposing an ``update`` method that accepts the detections and
        returns tracking results.
    """
    for frame in stream_frames(video_source):
        detections = detect_objects(frame)
        tracks = tracking.update(detections)
        yield frame, tracks


# Additional processing functions (canny_edge_detection, morph_dilate, etc.) could be
# added here as needed. The original project contained many more utilities; they were
# trimmed for brevity in this refactor.
