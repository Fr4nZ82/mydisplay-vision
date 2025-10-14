from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import cv2

BBox = Tuple[int, int, int, int]
Det = Tuple[BBox, float]  # ((x, y, w, h), score)

class YuNetDetector:
    """
    Wrapper per OpenCV FaceDetectorYN (YuNet).
    - detect(): ritorna lista di ((x,y,w,h), score)
    - detect_with_kps(): ritorna lista di ((x,y,w,h), score, kps5[5x2])
    """
    def __init__(self, model_path: str, score_th: float = 0.7, nms_iou: float = 0.3,
                 top_k: int = 500, backend_id: int = 0, target_id: int = 0) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Modello YuNet non trovato: {model_path}\n"
                "Assicurati che esista e che il path in config.detector_model sia corretto."
            )
        # input_size verrà impostata dinamicamente sul frame
        self.detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), score_th, nms_iou, top_k, backend_id, target_id
        )
        self._cur_size = (0, 0)

    def _ensure_size(self, w: int, h: int) -> None:
        if (w, h) != self._cur_size:
            self.detector.setInputSize((w, h))
            self._cur_size = (w, h)

    def detect(self, frame_bgr: np.ndarray) -> List[Det]:
        """
        :param frame_bgr: immagine BGR (uint8)
        :return: lista [ ((x,y,w,h), score), ... ]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        self._ensure_size(w, h)
        # FaceDetectorYN accetta BGR
        # detect() ritorna (num, dets) dove dets è Nx15: [x,y,w,h, 5*(lx,ly), score]
        _, dets = self.detector.detect(frame_bgr)
        if dets is None:
            return []
        out: List[Det] = []
        for d in dets:
            x, y, bw, bh = d[0:4]
            score = float(d[-1])
            out.append(((int(x), int(y), int(bw), int(bh)), score))
        return out

    def detect_with_kps(self, frame_bgr: np.ndarray) -> List[Tuple[Det, np.ndarray]]:
        """
        :param frame_bgr: immagine BGR (uint8)
        :return: lista [ ( ((x,y,w,h), score), kps5[5x2] ), ... ]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        self._ensure_size(w, h)
        _, dets = self.detector.detect(frame_bgr)
        if dets is None:
            return []
        out: List[Tuple[Det, np.ndarray]] = []
        for d in dets:
            x, y, bw, bh = d[0:4]
            score = float(d[-1])
            # d[4:14] contiene 5 coppie (lx, ly)
            kps = np.array([
                [d[4], d[5]],
                [d[6], d[7]],
                [d[8], d[9]],
                [d[10], d[11]],
                [d[12], d[13]],
            ], dtype=np.float32)
            out.append((((int(x), int(y), int(bw), int(bh)), score), kps))
        return out

