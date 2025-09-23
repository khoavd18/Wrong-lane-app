# src/wrong_lane/overlay.py
# -*- coding: utf-8 -*-
from typing import Dict, List
import cv2
import numpy as np
from .lanes import poly_to_np
from .colors import (COLOR_MOTO_FILL, COLOR_OTHER_FILL, COLOR_EDGE,
                     COLOR_OK_CAR, COLOR_OK_MOTO, COLOR_WRONG, COLOR_TEXT)

def draw_lane_overlay(base_img, lanes: List[Dict], fill_alpha: float, poly_thick: int):
    overlay = base_img.copy()
    for lane in lanes:
        fill_color = COLOR_MOTO_FILL if lane['type'] == 'moto' else COLOR_OTHER_FILL
        pts, pts2 = poly_to_np(lane['poly'])  # type: ignore
        mask = np.zeros_like(base_img)
        cv2.fillPoly(mask, [pts], fill_color)
        overlay = cv2.addWeighted(mask, fill_alpha, overlay, 1.0 - fill_alpha, 0.0)
        cv2.polylines(overlay, [pts], True, COLOR_EDGE, poly_thick, cv2.LINE_AA)

        M = cv2.moments(pts2)
        if abs(M['m00']) > 1e-5:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            txt = 'MOTORCYCLE' if lane['type'] == 'moto' else 'OTHER'
            cv2.putText(overlay, txt, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    return overlay

def draw_detection(out, x1,y1,x2,y2, cls: str, wrong: bool):
    color = COLOR_WRONG if wrong else (COLOR_OK_MOTO if cls in {'motorbike','motorcycle','bicycle'} else COLOR_OK_CAR)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if wrong else 2, cv2.LINE_AA)
    label = f'{cls}' + (' WRONG' if wrong else '')
    cv2.putText(out, label, (x1, max(20, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.circle(out, ((x1+x2)//2, y2-3), 4, (0,255,255), -1)
