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

"""Kalman filter based multi-object tracker with Hungarian assignment.

This module implements a lightweight SORT-style tracker using a constant
velocity Kalman filter and the Hungarian algorithm for data association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Detection:
    """A single object detection represented by a bounding box."""

    bbox: np.ndarray  # (x1, y1, x2, y2)
    score: float = 1.0


@dataclass
class Track:
    """Internal representation of a tracked object."""

    id: int
    mean: np.ndarray
    covariance: np.ndarray
    hits: int
    misses: int

    @property
    def bbox(self) -> np.ndarray:
        """Current bounding box in (x1, y1, x2, y2) format."""
        return _state_to_bbox(self.mean)


class KalmanFilter:
    """Simple constant velocity Kalman filter for bounding boxes."""

    def __init__(self) -> None:
        dt = 1.0
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = dt
        self._H = np.zeros((4, 8))
        self._H[:4, :4] = np.eye(4)
        self._Q = np.eye(8) * 0.01
        self._R = np.eye(4)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(8)
        mean[:4] = measurement
        covariance = np.eye(8)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + self._Q
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        S = self._H @ covariance @ self._H.T + self._R
        K = covariance @ self._H.T @ np.linalg.inv(S)
        innovation = measurement - (self._H @ mean)
        mean = mean + K @ innovation
        covariance = (np.eye(8) - K @ self._H) @ covariance
        return mean, covariance


def _bbox_to_state(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.array([cx, cy, w, h])


def _state_to_bbox(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2])


def iou_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    """Compute IoU matrix between tracks and detections."""
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)))
    track_boxes = np.array([t.bbox for t in tracks])
    det_boxes = np.array([d.bbox for d in detections])

    # Convert to (x1,y1,x2,y2)
    iou = np.zeros((len(tracks), len(detections)))
    for t, tb in enumerate(track_boxes):
        tx1, ty1, tx2, ty2 = tb
        t_area = (tx2 - tx1) * (ty2 - ty1)
        for d, db in enumerate(det_boxes):
            dx1, dy1, dx2, dy2 = db
            inter_x1 = max(tx1, dx1)
            inter_y1 = max(ty1, dy1)
            inter_x2 = min(tx2, dx2)
            inter_y2 = min(ty2, dy2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                det_area = (dx2 - dx1) * (dy2 - dy1)
                iou[t, d] = inter_area / float(t_area + det_area - inter_area)
            else:
                iou[t, d] = 0.0
    return iou


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve the linear assignment problem using the Hungarian algorithm."""
    cost = cost_matrix.copy()
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T
        n, m = cost.shape
        transposed = True
    u = np.zeros(n)
    v = np.zeros(m)
    ind = np.full(m, -1, dtype=int)

    for i in range(n):
        links = np.full(m, -1, dtype=int)
        mins = np.full(m, np.inf)
        visited = np.zeros(m, dtype=bool)
        marked_i = i
        marked_j = -1
        j = 0
        while True:
            j = -1
            for j1 in range(m):
                if not visited[j1]:
                    cur = cost[marked_i, j1] - u[marked_i] - v[j1]
                    if cur < mins[j1]:
                        mins[j1] = cur
                        links[j1] = marked_j
                    if j == -1 or mins[j1] < mins[j]:
                        j = j1
            delta = mins[j]
            for j1 in range(m):
                if visited[j1]:
                    u[ind[j1]] += delta
                    v[j1] -= delta
                else:
                    mins[j1] -= delta
            u[i] += delta
            visited[j] = True
            marked_j = j
            marked_i = ind[j]
            if marked_i == -1:
                break
        while True:
            if links[j] != -1:
                ind[j] = ind[links[j]]
                j = links[j]
            else:
                break
        ind[j] = i
    if transposed:
        return np.array([[ind[j], j] for j in range(m) if ind[j] >= 0])[:, ::-1]
    return np.array([[ind[j], j] for j in range(m) if ind[j] >= 0])


class SortTracker:
    """Simple SORT-like multi-object tracker."""

    def __init__(self, max_age: int = 1, iou_threshold: float = 0.3) -> None:
        self.kf = KalmanFilter()
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections and return active tracks."""
        # Predict existing tracks
        for track in self.tracks:
            track.mean, track.covariance = self.kf.predict(track.mean, track.covariance)

        if detections:
            measurements = np.array([_bbox_to_state(d.bbox) for d in detections])
        else:
            measurements = np.empty((0, 4))

        # Associate detections to tracks
        if self.tracks and detections:
            iou = iou_matrix(self.tracks, detections)
            cost_matrix = 1.0 - iou
            matches = linear_assignment(cost_matrix)

            unmatched_tracks = set(range(len(self.tracks)))
            unmatched_dets = set(range(len(detections)))
            for t_idx, d_idx in matches:
                if iou[t_idx, d_idx] < self.iou_threshold:
                    unmatched_tracks.add(t_idx)
                    unmatched_dets.add(d_idx)
                    continue
                unmatched_tracks.discard(t_idx)
                unmatched_dets.discard(d_idx)
                track = self.tracks[t_idx]
                track.mean, track.covariance = self.kf.update(
                    track.mean, track.covariance, measurements[d_idx]
                )
                track.hits += 1
                track.misses = 0

            for t_idx in unmatched_tracks:
                self.tracks[t_idx].misses += 1

            for d_idx in unmatched_dets:
                mean, cov = self.kf.initiate(measurements[d_idx])
                self.tracks.append(
                    Track(self._next_id, mean, cov, hits=1, misses=0)
                )
                self._next_id += 1
        else:
            for track in self.tracks:
                track.misses += 1
            for det in detections:
                mean, cov = self.kf.initiate(_bbox_to_state(det.bbox))
                self.tracks.append(Track(self._next_id, mean, cov, hits=1, misses=0))
                self._next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [t for t in self.tracks]


__all__ = ["Detection", "Track", "SortTracker"]
