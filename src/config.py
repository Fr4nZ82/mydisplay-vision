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

    @staticmethod
    def load(path: str) -> "AppConfig":
        cfg = AppConfig()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except FileNotFoundError:
            pass
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
