# src/wrong_lane/yolo_runner.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
import os
from ultralytics import YOLO
from .utils import pick_device, get_class_name
from .config import VALID_CLASSES

class YoloRunner:
    """Bọc YOLOv8 + lọc ngưỡng theo lớp (CONF_PER_CLASS) với fallback an toàn."""
    def __init__(self, weights: str, device_str: str, imgsz: int,
                 per_thr: Dict[str, float], default_thr: float):
        self.device = pick_device(device_str)
        self.imgsz = int(imgsz)
        self.per_thr = dict(per_thr or {})
        self.default_thr = float(default_thr)

        w = weights
        try:
            if os.path.isfile(w) and os.path.getsize(w) < 1_000_000:
                print(f"[WARN] Weights '{w}' co dung luong bat thuong ({os.path.getsize(w)} bytes). "
                      f"Fallback sang 'yolov8n.pt' de Ultralytics tu tai.")
                w = "yolov8n.pt"
            self.model = YOLO(w)
        except Exception as e:
            if os.path.isfile(weights):
                print(f"[WARN] Khong load duoc weights '{weights}': {e}\n-> Fallback 'yolov8n.pt'")
                self.model = YOLO("yolov8n.pt")
            else:
                raise

    def detect(self, frame) -> Tuple[List[List[int]], List[str]]:
        min_conf = min(self.per_thr.values()) if self.per_thr else self.default_thr
        res = self.model.predict(frame, imgsz=self.imgsz, conf=min_conf, iou=0.5,
                                 device=self.device, verbose=False)
        dets_xyxy, det_classes = [], []
        if len(res):
            r = res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                names = self.model.names
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(b.cls[0].cpu().numpy())
                    cls = get_class_name(names, cls_id)
                    if cls not in VALID_CLASSES:
                        continue
                    conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else 1.0
                    need = float(self.per_thr.get(cls, self.default_thr))
                    if conf < need:
                        continue
                    dets_xyxy.append([int(x1), int(y1), int(x2), int(y2)])
                    det_classes.append(cls)
        return dets_xyxy, det_classes
