# -*- coding: utf-8 -*-
"""
SORT-lite (no deps): IoU matching (greedy), track lifecycle, stable IDs.
Includes per-track smoothing for gender/age via small history window + EMA.
Inputs per frame: list of detections as [x, y, w, h, score].
Outputs: list of tracks dict:
    {
      'track_id': int,
      'bbox': [x, y, w, h],
      'score': float,
      'hits': int,
      'age': int,        # frames since last update
      'gender': 'male'|'female'|'unknown',
      'ageBucket': '0-13'|...|'unknown',
      'conf': float      # smoothed confidence
    }
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from collections import deque, Counter
import itertools
import math

def iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / (area_a + area_b - inter + 1e-9)


class _LabelSmoother:
    """
    Keeps a short window of labels, majority vote + EMA confidence.
    """
    def __init__(self, win: int = 8, ema_alpha: float = 0.5):
        self.gender_hist = deque(maxlen=win)
        self.age_hist = deque(maxlen=win)
        self.conf_ema: float = 0.0
        self.alpha = float(ema_alpha)

    def update(self, gender: str, age_bucket: str, conf: float) -> Tuple[str, str, float]:
        if gender:
            self.gender_hist.append(gender)
        if age_bucket:
            self.age_hist.append(age_bucket)
        self.conf_ema = self.alpha * float(conf) + (1.0 - self.alpha) * float(self.conf_ema)
        g = self._majority(self.gender_hist, default="unknown")
        a = self._majority(self.age_hist, default="unknown")
        return g, a, float(self.conf_ema)

    @staticmethod
    def _majority(hist: deque, default: str) -> str:
        if not hist:
            return default
        c = Counter(hist)
        lab, cnt = c.most_common(1)[0]
        return lab if cnt > 0 else default


class SortLiteTracker:
    def __init__(
        self,
        max_age: int = 20,
        min_hits: int = 2,
        iou_th: float = 0.3,
        smooth_win: int = 8,
        smooth_alpha: float = 0.5,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_th = float(iou_th)
        self.tracks: Dict[int, Dict] = {}
        self._next_id = 1
        self.smooth_win = int(smooth_win)
        self.smooth_alpha = float(smooth_alpha)

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def _new_track(self, det) -> Dict:
        tid = self._next_id
        self._next_id += 1
        x, y, w, h, s = det
        tr = {
            "track_id": tid,
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": float(s),
            "hits": 1,
            "age": 0,
            "time_since_update": 0,
            "smoother": _LabelSmoother(win=self.smooth_win, ema_alpha=self.smooth_alpha),
            "gender": "unknown",
            "ageBucket": "unknown",
            "conf": 0.0,
            "prev_center": self._center([x, y, w, h]),
        }
        self.tracks[tid] = tr
        return tr

    @staticmethod
    def _center(b: List[float]) -> Tuple[float, float]:
        return (b[0] + b[2] * 0.5, b[1] + b[3] * 0.5)

    def update(self, detections: List[List[float]]) -> List[Dict]:
        """
        detections: list of [x,y,w,h,score]
        returns: active tracks
        """
        # aging
        for tr in self.tracks.values():
            tr["age"] += 1
            tr["time_since_update"] += 1

        # Build IoU matrix (tracks x detections)
        tr_ids = list(self.tracks.keys())
        unmatched_tr = set(tr_ids)
        unmatched_det = set(range(len(detections)))
        matches = []

        if tr_ids and detections:
            iou_mat = [[0.0 for _ in range(len(detections))] for _ in tr_ids]
            for i, tid in enumerate(tr_ids):
                tb = self.tracks[tid]["bbox"]
                for j, det in enumerate(detections):
                    iou_mat[i][j] = iou_xywh(tb, det[:4])

            # Greedy matching by best IoU above threshold
            taken_det = set()
            for i, tid in sorted(enumerate(tr_ids), key=lambda x: x[0]):
                # find best detection for this track
                best_j, best_iou = -1, 0.0
                for j in range(len(detections)):
                    if j in taken_det:
                        continue
                    if iou_mat[i][j] > best_iou:
                        best_iou = iou_mat[i][j]
                        best_j = j
                if best_j >= 0 and best_iou >= self.iou_th:
                    matches.append((tid, best_j))
                    taken_det.add(best_j)

            unmatched_tr = set(tr_ids) - {tid for tid, _ in matches}
            unmatched_det = set(range(len(detections))) - {j for _, j in matches}

        # Update matched tracks
        for tid, j in matches:
            det = detections[j]
            tr = self.tracks[tid]
            tr["bbox"] = [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
            tr["score"] = float(det[4])
            tr["hits"] += 1
            tr["age"] = 0
            tr["time_since_update"] = 0

        # Create new tracks for unmatched detections
        for j in unmatched_det:
            self._new_track(detections[j])

        # Remove dead tracks
        to_delete = [tid for tid, tr in self.tracks.items() if tr["time_since_update"] > self.max_age]
        for tid in to_delete:
            del self.tracks[tid]

        # Return active tracks (min_hits gating)
        out = []
        for tid, tr in self.tracks.items():
            if tr["hits"] >= self.min_hits or tr["time_since_update"] == 0:
                out.append({
                    "track_id": tid,
                    "bbox": list(tr["bbox"]),
                    "score": float(tr["score"]),
                    "hits": int(tr["hits"]),
                    "age": int(tr["age"]),
                    "gender": tr.get("gender", "unknown"),
                    "ageBucket": tr.get("ageBucket", "unknown"),
                    "conf": float(tr.get("conf", 0.0)),
                    "prev_center": tr.get("prev_center", self._center(tr["bbox"])),
                })
        return out

    def apply_labels(self, track_id: int, gender: str, age_bucket: str, conf: float):
        """
        Update smoothed labels for a given track.
        """
        tr = self.tracks.get(track_id)
        if not tr:
            return
        g, a, c = tr["smoother"].update(gender, age_bucket, conf)
        tr["gender"], tr["ageBucket"], tr["conf"] = g, a, c

    def update_prev_center(self, track_id: int):
        tr = self.tracks.get(track_id)
        if tr:
            tr["prev_center"] = self._center(tr["bbox"])
