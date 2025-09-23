# src/wrong_lane/utils.py
# -*- coding: utf-8 -*-
from typing import Any
import cv2
import torch

def pick_device(dev_str: str) -> str:
    if dev_str and dev_str.lower() != 'auto':
        return dev_str
    return '0' if torch.cuda.is_available() else 'cpu'

def get_class_name(names: Any, cid: int) -> str:
    if isinstance(names, (list, tuple)):
        return names[cid] if 0 <= cid < len(names) else str(cid)
    return names.get(cid, str(cid))

def put_hud(out, fps_ema: float, title: str) -> None:
    cv2.putText(out, f'FPS: {fps_ema:.2f}', (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), 2, cv2.LINE_AA)
