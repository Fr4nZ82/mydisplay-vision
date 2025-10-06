from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class AppConfig:
    camera: int = 0
    width: int = 640
    height: int = 480
    target_fps: float = 10.0
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    # --- debug options ---
    debug_enabled: bool = True
    debug_stream_fps: float = 5.0
    debug_resize_width: int = 720
    # --- detector (YuNet) ---
    detector_model: str = "models/face_detection_yunet_2023mar.onnx"
    detector_score_th: float = 0.7
    detector_nms_iou: float = 0.3
    detector_top_k: int = 500
    detector_backend: int = 0     # 0=default (CPU)
    detector_target: int = 0      # 0=CPU

    @staticmethod
    def load(path: str) -> "AppConfig":
        cfg = AppConfig()
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                data: dict[str, Any] = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except FileNotFoundError:
            pass
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
