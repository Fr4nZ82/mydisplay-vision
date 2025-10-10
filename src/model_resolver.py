# -*- coding: utf-8 -*-
"""
model_resolver.py
-----------------
Risoluzione automatica dei modelli da:
models/
  face/
  person/
  genderage/
  reid_face/
  reid_body/
ognuna con sottocartelle 'openvino/' (IR .xml+.bin) e/o 'onnx/' (.onnx).
Precedenza: openvino -> onnx.
"""

from __future__ import annotations
import os, glob
from typing import Dict, Optional

CATEGORIES = ("face", "person", "genderage", "reid_face", "reid_body")

def _pick_openvino(dirpath: str) -> Dict[str, str]:
    xmls = sorted(glob.glob(os.path.join(dirpath, "*.xml")))
    for xml in xmls:
        binp = os.path.splitext(xml)[0] + ".bin"
        if os.path.exists(binp):
            return {"kind": "openvino", "xml": xml, "bin": binp, "onnx": ""}
    return {"kind": "none", "xml": "", "bin": "", "onnx": ""}

def _pick_onnx(dirpath: str) -> Dict[str, str]:
    onnxs = sorted(glob.glob(os.path.join(dirpath, "*.onnx")))
    if onnxs:
        return {"kind": "onnx", "xml": "", "bin": "", "onnx": onnxs[0]}
    return {"kind": "none", "xml": "", "bin": "", "onnx": ""}

def resolve_category(category: str, models_root: str = "models") -> Dict[str, str]:
    """
    Ritorna dict con chiavi: kind, xml, bin, onnx, exists (bool), base.
    Precedenza: openvino/ → onnx/.
    """
    out = {"kind": "none", "xml": "", "bin": "", "onnx": "", "exists": False, "base": ""}
    if category not in CATEGORIES:
        return out
    base = os.path.join(models_root, category)
    out["base"] = base
    if not os.path.isdir(base):
        return out

    # 1) openvino/
    ov = _pick_openvino(os.path.join(base, "openvino"))
    if ov["kind"] == "openvino":
        out.update(ov)
        out["exists"] = True
        return out

    # 2) onnx/
    on = _pick_onnx(os.path.join(base, "onnx"))
    if on["kind"] == "onnx":
        out.update(on)
        out["exists"] = True
        return out

    return out

def resolve_all(models_root: str = "models") -> Dict[str, Dict[str, str]]:
    return {c: resolve_category(c, models_root) for c in CATEGORIES}
