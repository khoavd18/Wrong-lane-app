# src/wrong_lane/config.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, Tuple, Set

CONFIG: Dict[str, Any] = {
    # Video mặc định (sẽ bị ghi đè nếu upload trên Streamlit)
    "VIDEO": r"assets/video_input/phamvandong.mp4",

    "LANE_JSON": "lanes_phamvandong1.json", # File JSON mặc định (sẽ bị ghi đè nếu upload trên Streamlit)
    "LANE_JSON_PATH": None,   # lưu JSON upload

    "YOLO_WEIGHTS": r"assets/yolov8n.pt",

    "DEVICE": "auto",
    "IMG_SIZE": 640,
    "CONF": 0.35,
    "CONF_PER_CLASS": {
        "car": 0.70,
        "truck": 0.55,
        "bus": 0.55,
        "motorcycle": 0.20,
        "motorbike": 0.25,
        "bicycle": 0.70
    },

    "TITLE": "LANE CONFIG: AUTO-LOAD or DRAW NEW (Per-class CONF)",
    "FPS_ALPHA": 0.9,
    "POLY_THICK": 3,
    "FILL_ALPHA": 0.28,
    "IN_REGION_MARGIN": 8,
    "REQUIRE_IN_LANE": True,
    "NEUTRAL_COLOR": (180, 180, 0),
    "LANE_DIR": "config",
    "AUTO_JSON_PER_VIDEO": True,
    
    "START_MODE": "ask",
}

VALID_CLASSES: Set[str] = {"car", "truck", "bus", "bicycle", "motorcycle", "motorbike"}
NEUTRAL_COLOR: Tuple[int, int, int] = CONFIG.get("NEUTRAL_COLOR", (180, 180, 0))  # type: ignore
