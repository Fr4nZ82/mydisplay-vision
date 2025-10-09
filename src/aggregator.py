# -*- coding: utf-8 -*-
"""
MinuteAggregator: conta CROSS sulla tripwire (linea o banda) e produce aggregati
per finestre di N secondi. Mantiene gli ultimi M minuti in memoria.
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from collections import deque, defaultdict
import time
import math


def _side_of_line(ax, ay, bx, by, px, py) -> float:
    """
    Signed area (cross product) to know which side of the directed line AB the point P lies on.
    >0: left, <0: right, 0: on the line.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _segment_intersects_band(ax, ay, bx, by, p0, p1, band_px: float) -> bool:
    """
    Rough check: distance of segment midpoint to line < band threshold OR opposite sides.
    """
    (x0, y0), (x1, y1) = p0, p1
    # if crossed sides, consider it a valid crossing
    s0 = _side_of_line(ax, ay, bx, by, x0, y0)
    s1 = _side_of_line(ax, ay, bx, by, x1, y1)
    if s0 == 0.0 or s1 == 0.0 or (s0 > 0 and s1 < 0) or (s0 < 0 and s1 > 0):
        return True
    # otherwise check band proximity at midpoint
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    # distance from point to line AB
    vx, vy = bx - ax, by - ay
    wx, wy = mx - ax, my - ay
    vlen = math.hypot(vx, vy) + 1e-9
    cross = abs(vx * wy - vy * wx)
    dist = cross / vlen
    return dist <= band_px * 0.5


class MinuteAggregator:
    def __init__(self, window_sec: int = 60, retention_min: int = 120):
        self.window_sec = int(window_sec)
        self.retention_min = int(retention_min)
        self.windows = deque()   # list of dict payloads
        self.current_win_start = None
        self._count_map = defaultdict(int)
        self._age_map = defaultdict(int)
        self._already_crossed = set()  # (track_id, dir_tag, win_start) to avoid duplicates

    def _roll_window(self, now: float):
        if self.current_win_start is None:
            self.current_win_start = int(now // self.window_sec) * self.window_sec
            return
        # if we advanced to a new window
        target_start = int(now // self.window_sec) * self.window_sec
        if target_start != self.current_win_start:
            # flush previous window
            self._flush_current_window()
            self.current_win_start = target_start
            self._count_map.clear()
            self._age_map.clear()
            # clean up retention
            cutoff = now - (self.retention_min * 60)
            while self.windows:
                head = self.windows[0]
                epoch = head.get("_epoch")   # <-- non crasherà se manca
                if epoch is None or epoch >= cutoff:
                    break
                self.windows.popleft()

    def _flush_current_window(self):
        if self.current_win_start is None:
            return
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.current_win_start))
        payload = {
            "ts": ts_iso,
            "windowSec": self.window_sec,
            "counts": {
                "total": int(self._count_map.get("total", 0)),
                "male": int(self._count_map.get("male", 0)),
                "female": int(self._count_map.get("female", 0)),
                "unknown": int(self._count_map.get("unknown", 0)),
            },
            "ageBuckets": {  # include unknown if present
                k: int(v) for k, v in sorted(self._age_map.items(), key=lambda kv: kv[0])
            },
            "_epoch": int(self.current_win_start),
        }
        # ensure all buckets are present (even if 0)
        for b in ["0-13", "14-24", "25-34", "35-44", "45-54", "55-64", "65+", "unknown"]:
            payload["ageBuckets"].setdefault(b, 0)
        self.windows.append(payload)

    def add_cross_event(self, gender: str, age_bucket: str, direction_tag: str, track_id: int, now: Optional[float] = None):
        now = time.time() if now is None else now
        self._roll_window(now)

        # prevent duplicates in the same window/direction
        dedup_key = (int(track_id), direction_tag, int(self.current_win_start))
        if dedup_key in self._already_crossed:
            return
        self._already_crossed.add(dedup_key)

        self._count_map["total"] += 1
        if gender in ("male", "female"):
            self._count_map[gender] += 1
        else:
            self._count_map["unknown"] += 1

        if age_bucket in ("0-13", "14-24", "25-34", "35-44", "45-54", "55-64", "65+"):
            self._age_map[age_bucket] += 1
        else:
            self._age_map["unknown"] += 1

    def get_last(self, n: int = 10) -> List[Dict]:
      # Ritorna gli ultimi n window COMPLETATI (senza mutare l'originale)
      src = list(self.windows)[-n:] if n > 0 else list(self.windows)
      out: List[Dict] = []
      for w in src:
          wc = dict(w)             # <-- copia shallow del dict
          wc.pop("_epoch", None)   # <-- rimuovi solo nella copia
          out.append(wc)
      return out
    
    def get_current(self) -> Optional[Dict]:
        """Snapshot della finestra corrente (non finalizzata)."""
        if self.current_win_start is None:
            return None
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.current_win_start))
        payload = {
            "ts": ts_iso,
            "windowSec": self.window_sec,
            "counts": {
                "total": int(self._count_map.get("total", 0)),
                "male": int(self._count_map.get("male", 0)),
                "female": int(self._count_map.get("female", 0)),
                "unknown": int(self._count_map.get("unknown", 0)),
            },
            "ageBuckets": {k: int(v) for k, v in sorted(self._age_map.items(), key=lambda kv: kv[0])},
            "_epoch": int(self.current_win_start),
            "_current": True,
        }
        for b in ["0-13","14-24","25-34","35-44","45-54","55-64","65+","unknown"]:
            payload["ageBuckets"].setdefault(b, 0)
        return payload

    def tick(self, now: Optional[float] = None) -> None:
        """Fa avanzare le finestre anche senza eventi."""
        self._roll_window(time.time() if now is None else now)

    # ---- Tripwire check helpers ----

    @staticmethod
    def check_crossing(
        frame_w: int,
        frame_h: int,
        roi_tripwire: Tuple[Tuple[float, float], Tuple[float, float]],
        roi_band_px: int,
        roi_direction: str,
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
    ) -> Optional[str]:
        """
        Returns direction tag 'a2b'|'b2a' if valid per roi_direction, else None.
        """
        (axn, ayn), (bxn, byn) = roi_tripwire
        ax, ay = axn * frame_w, ayn * frame_h
        bx, by = bxn * frame_w, byn * frame_h
        if not _segment_intersects_band(ax, ay, bx, by, prev_center, curr_center, float(roi_band_px)):
            return None

        s0 = _side_of_line(ax, ay, bx, by, prev_center[0], prev_center[1])
        s1 = _side_of_line(ax, ay, bx, by, curr_center[0], curr_center[1])
        if s0 == 0.0 or s1 == 0.0:
            return None  # ambiguous

        dir_tag = "a2b" if (s0 > 0 and s1 < 0) else ("b2a" if (s0 < 0 and s1 > 0) else None)
        if dir_tag is None:
            return None

        if roi_direction == "both":
            return dir_tag
        if roi_direction == dir_tag:
            return dir_tag
        return None
