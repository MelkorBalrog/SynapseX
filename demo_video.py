#!/usr/bin/env python3
"""Simple demo showcasing ``synapse.utils.draw_tracks``."""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2

from synapse.utils import draw_tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate video with dummy tracks")
    parser.add_argument("input", type=Path, help="Path to input video")
    parser.add_argument("output", type=Path, help="Where to save annotated video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = (10 + frame_idx, 10 + frame_idx, 110 + frame_idx, 110 + frame_idx)
        tracks = [{"bbox": bbox, "label": "obj", "id": 0}]
        draw_tracks(frame, tracks)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
