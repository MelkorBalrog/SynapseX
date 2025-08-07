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

"""Demonstration of multi-object detection and tracking using SynapseX."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Dict, Tuple

import cv2
import numpy as np

from synapse.utils.image_processing import stream_tracks


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_objects(frame: np.ndarray):
    """Detect people in ``frame`` using OpenCV's built-in HOG detector."""
    rects, _ = _hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]


# ---------------------------------------------------------------------------
# Simple centroid tracker
# ---------------------------------------------------------------------------
class CentroidTracker:
    """Minimal centroid-based multi-object tracker."""

    def __init__(self, max_disappeared: int = 40):
        self.next_object_id = 0
        self.objects: "OrderedDict[int, Tuple[int, int, int, int]]" = OrderedDict()
        self.disappeared: "OrderedDict[int, int]" = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, bbox: Tuple[int, int, int, int]):
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2), int(y + h / 2))

        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = []
            for bbox in self.objects.values():
                (x, y, w, h) = bbox
                object_centroids.append((int(x + w / 2), int(y + h / 2)))
            object_centroids = np.array(object_centroids)

            distances = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = rects[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distances.shape[0])).difference(used_rows)
            unused_cols = set(range(distances.shape[1])).difference(used_cols)

            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(rects[col])

        return self.objects


# ---------------------------------------------------------------------------
# Demo execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-object tracking demo")
    parser.add_argument(
        "source",
        nargs="?",
        default=0,
        help="Video file path or camera index (default: 0)",
    )
    args = parser.parse_args()

    tracker = CentroidTracker()

    for frame, tracks in stream_tracks(args.source, detect_objects, tracker):
        for object_id, (x, y, w, h) in tracks.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(object_id),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("SynapseX Tracking Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

