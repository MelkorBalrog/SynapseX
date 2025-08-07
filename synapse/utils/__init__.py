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
"""Utility helpers for SynapseX."""

from __future__ import annotations

from typing import Iterable, Mapping, Any


def draw_tracks(frame, tracks: Iterable[Any]):
    """Draw tracking boxes, labels and IDs on ``frame``.

    Parameters
    ----------
    frame : numpy.ndarray
        Image to annotate in BGR format.  The drawing is performed in place.
    tracks : iterable
        Each element may be a mapping or an object exposing ``bbox``
        (``x1, y1, x2, y2``), ``id`` or ``track_id`` and an optional
        ``label`` attribute.
    """

    import cv2

    for trk in tracks:
        if isinstance(trk, Mapping):
            bbox = trk.get("bbox")
            track_id = trk.get("id", trk.get("track_id"))
            label = trk.get("label", "")
        else:
            bbox = getattr(trk, "bbox", None)
            track_id = getattr(trk, "id", getattr(trk, "track_id", None))
            label = getattr(trk, "label", "")

        if bbox is None or track_id is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {track_id}".strip()
        cv2.putText(
            frame,
            caption,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame

