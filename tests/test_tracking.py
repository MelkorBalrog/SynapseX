import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from synapse.tracking import Detection, SortTracker


def test_tracker_persistent_ids():
    tracker = SortTracker(max_age=0)
    # Frame 1: two detections
    f1 = [Detection(np.array([0, 0, 10, 10])), Detection(np.array([100, 100, 110, 110]))]
    tracks1 = tracker.update(f1)
    assert len(tracks1) == 2
    ids1 = [t.id for t in tracks1]

    # Frame 2: detections move slightly
    f2 = [Detection(np.array([1, 1, 11, 11])), Detection(np.array([101, 101, 111, 111]))]
    tracks2 = tracker.update(f2)
    ids2 = [t.id for t in tracks2]
    assert ids1 == ids2

    # Frame 3: second detection disappears
    f3 = [Detection(np.array([2, 2, 12, 12]))]
    tracks3 = tracker.update(f3)
    assert len(tracks3) == 1
    assert tracks3[0].id == ids1[0]
