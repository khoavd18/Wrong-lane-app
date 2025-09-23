# src/wrong_lane/lanes.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional
import json, os
import cv2
import numpy as np

def poly_to_np(poly4: List[Tuple[int, int]]):
    pts = np.array(poly4, dtype=np.int32).reshape(-1, 1, 2)
    pts2 = np.array(poly4, dtype=np.int32).reshape(-1, 2)
    return pts, pts2

def inside_with_margin(poly_np2, pt_xy: Tuple[int, int], delta: float = 8) -> bool:
    return cv2.pointPolygonTest(poly_np2, pt_xy, True) >= -float(delta)

def scale_poly(poly: List[Tuple[int, int]], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    w0, h0 = from_size
    w, h = to_size
    if w0 <= 0 or h0 <= 0 or (w0 == w and h0 == h):
        return poly
    sx, sy = float(w) / float(w0), float(h) / float(h0)
    return [(int(x * sx), int(y * sy)) for (x, y) in poly]

def save_lanes_json(path: str, lanes: List[Dict], meta: Optional[Dict] = None) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    data = {'lanes': lanes, 'meta': meta or {}}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_lanes_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def resolve_lane_json_path(cfg: Dict, video_path: str) -> str:
    lane_dir = str(cfg.get('LANE_DIR', 'config'))
    os.makedirs(lane_dir, exist_ok=True)
    use_per_video = bool(cfg.get('AUTO_JSON_PER_VIDEO', True))
    if use_per_video:
        base = os.path.splitext(os.path.basename(video_path))[0] or 'default'
        return os.path.join(lane_dir, f'lanes_{base}.json')
    name = str(cfg.get('LANE_JSON', 'lanes_config.json'))
    return name if os.path.isabs(name) else os.path.join(lane_dir, name)
