# src/wrong_lane/app.py
# -*- coding: utf-8 -*-
from typing import Dict, List
import cv2
import os

from .config import CONFIG
from .utils import put_hud
from .lanes import (
    load_lanes_json, save_lanes_json, scale_poly,
    poly_to_np, inside_with_margin, resolve_lane_json_path
)
from .overlay import draw_lane_overlay, draw_detection
from .ui_draw import draw_lanes_ui
from .yolo_runner import YoloRunner


def _ask_start_mode() -> str:
    mode = str(CONFIG.get('START_MODE', 'ask')).lower()
    if mode in ('auto', 'draw', 'json_custom'):
        return mode
    # Chỉ chạy khi không có Streamlit (terminal)
    print('\n=== CHON CHE DO KHOI DONG ===')
    print('1) Auto-load config (lanes JSON) neu co')
    print('2) Ve config moi (chon so lan -> bam 4 diem/lan)')
    print('3) Auto-load tu duong dan khac')
    s = input('Nhap 1/2/3 [mac dinh 1]: ').strip()
    if s == '2': return 'draw'
    if s == '3': return 'auto_custom'
    return 'auto'


def _load_or_draw_lanes(first_frame, lane_json: str):
    H, W = first_frame.shape[:2]
    mode = _ask_start_mode()
    lanes: List[Dict] = []

    if mode == 'auto':
        # Load tự động từ JSON theo tên video
        data = load_lanes_json(lane_json)
        if data and 'lanes' in data:
            meta = data.get('meta', {})
            size_meta = tuple(meta.get('size', [W, H]))
            lanes_raw = data['lanes']
            if CONFIG.get('ALLOW_SIZE_RESCALE', True) and size_meta:
                lanes = []
                for ln in lanes_raw:
                    ln2 = dict(ln)
                    ln2['poly'] = scale_poly(ln['poly'], (size_meta[0], size_meta[1]), (W, H))  # type: ignore
                    lanes.append(ln2)
            else:
                lanes = lanes_raw
            print(f'[OK] Auto-loaded {len(lanes)} lanes from {lane_json}')
        else:
            print('[WARN] Khong co JSON. Chuyen sang ve.')
            lanes = draw_lanes_ui(first_frame, lane_json,
                                  fill_alpha=float(CONFIG['FILL_ALPHA']),
                                  poly_thick=int(CONFIG['POLY_THICK'])) or []

    elif mode == 'json_custom':
        # Load JSON do người dùng upload (qua Streamlit)
        path = CONFIG.get('LANE_JSON_PATH')
        if not path:
            print('[ERR] Chưa có file JSON lane từ Streamlit hoặc cấu hình.')
            return []
        data = load_lanes_json(path)
        if data and 'lanes' in data:
            meta = data.get('meta', {})
            size_meta = tuple(meta.get('size', [W, H]))
            lanes_raw = data['lanes']
            if CONFIG.get('ALLOW_SIZE_RESCALE', True) and size_meta:
                lanes = []
                for ln in lanes_raw:
                    ln2 = dict(ln)
                    ln2['poly'] = scale_poly(ln['poly'], (size_meta[0], size_meta[1]), (W, H))  # type: ignore
                    lanes.append(ln2)
            else:
                lanes = lanes_raw
            print(f'[OK] Auto-loaded {len(lanes)} lanes from {path}')
            # Lưu lại dưới tên mặc định để lần sau auto-load được
            save_lanes_json(lane_json, lanes, meta={'size': [W, H]})
        else:
            print('[WARN] Khong load duoc file JSON chi dinh. Chuyen sang ve.')
            lanes = draw_lanes_ui(first_frame, lane_json,
                                  fill_alpha=float(CONFIG['FILL_ALPHA']),
                                  poly_thick=int(CONFIG['POLY_THICK'])) or []

    else:  # draw
        lanes = draw_lanes_ui(first_frame, lane_json,
                              fill_alpha=float(CONFIG['FILL_ALPHA']),
                              poly_thick=int(CONFIG['POLY_THICK'])) or []

    return lanes


def main():
    video_path = os.environ.get("VIDEO") or str(CONFIG['VIDEO'])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {CONFIG['VIDEO']}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError('Không đọc được frame đầu tiên.')
    H, W = first_frame.shape[:2]

    lane_json = resolve_lane_json_path(CONFIG, str(CONFIG['VIDEO']))
    lanes = _load_or_draw_lanes(first_frame, lane_json)
    if not lanes:
        cap.release()
        cv2.destroyAllWindows()
        return

    runner = YoloRunner(weights=str(CONFIG['YOLO_WEIGHTS']),
                        device_str=str(CONFIG['DEVICE']),
                        imgsz=int(CONFIG['IMG_SIZE']),
                        per_thr=dict(CONFIG.get('CONF_PER_CLASS', {})),
                        default_thr=float(CONFIG.get('CONF', 0.35)))

    win = 'Wrong-lane (Run)'
    cv2.namedWindow(win)
    fps_ema = 0.0
    t_prev = float(cv2.getTickCount()) / cv2.getTickFrequency()

    require_in_lane = bool(CONFIG.get('REQUIRE_IN_LANE', True))
    neutral_color = CONFIG.get('NEUTRAL_COLOR', None)
    fill_alpha = float(CONFIG['FILL_ALPHA'])
    poly_thick = int(CONFIG['POLY_THICK'])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        dets_xyxy, det_classes = runner.detect(frame)

        overlay = draw_lane_overlay(frame, lanes, fill_alpha=fill_alpha, poly_thick=poly_thick)
        out = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        margin = max(int(CONFIG['IN_REGION_MARGIN']), int(0.01 * W))
        for (x1, y1, x2, y2), cls in zip(dets_xyxy, det_classes):
            foot = ((x1 + x2) // 2, y2 - 3)
            in_any = False
            in_moto = False
            for lane in lanes:
                _, poly2 = poly_to_np(lane['poly'])  # type: ignore
                if inside_with_margin(poly2, foot, delta=margin):
                    in_any = True
                    in_moto = (lane['type'] == 'moto')
                    break

            if require_in_lane and not in_any:
                if isinstance(neutral_color, (tuple, list)) and len(neutral_color) == 3:
                    nc = tuple(int(c) for c in neutral_color)  # type: ignore
                    cv2.rectangle(out, (x1, y1), (x2, y2), nc, 2, cv2.LINE_AA)
                    cv2.circle(out, foot, 4, nc, -1)
                continue

            is_bike = cls in {'motorbike', 'motorcycle', 'bicycle'}
            is_car = cls in {'car', 'truck', 'bus'}
            wrong = (is_bike and not in_moto) or (is_car and in_moto)
            draw_detection(out, x1, y1, x2, y2, cls, wrong)

        now = float(cv2.getTickCount()) / cv2.getTickFrequency()
        dt = now - t_prev
        t_prev = now
        fps_i = 1.0 / max(1e-6, dt)
        fps_ema = float(CONFIG['FPS_ALPHA']) * fps_ema + (1 - float(CONFIG['FPS_ALPHA'])) * fps_i
        put_hud(out, fps_ema, str(CONFIG['TITLE']))

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            save_lanes_json(lane_json, lanes, meta={'size': [W, H]})
            print(f'Da luu {lane_json}')
        elif key == ord('l'):
            data = load_lanes_json(lane_json)
            if data and 'lanes' in data:
                meta = data.get('meta', {})
                size_meta = tuple(meta.get('size', [W, H]))
                lanes_raw = data['lanes']
                if CONFIG.get('ALLOW_SIZE_RESCALE', True) and size_meta:
                    lanes = []
                    for ln in lanes_raw:
                        ln2 = dict(ln)
                        ln2['poly'] = scale_poly(ln['poly'], (size_meta[0], size_meta[1]), (W, H))  # type: ignore
                        lanes.append(ln2)
                else:
                    lanes = lanes_raw
                print(f'Da load {lane_json} ({len(lanes)} lanes)')
        elif key == ord('r'):
            lanes2 = draw_lanes_ui(first_frame, lane_json,
                                   fill_alpha=fill_alpha, poly_thick=poly_thick)
            if lanes2 is None:
                break
            lanes = lanes2

        # Kiểm tra nút X (close window)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    #####################################################################################

import streamlit as st

def main_streamlit():
    import streamlit as st

    video_path = os.environ.get("VIDEO") or str(CONFIG['VIDEO'])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Không mở được video: {CONFIG['VIDEO']}")
        return

    ok, first_frame = cap.read()
    if not ok:
        st.error("Không đọc được frame đầu tiên.")
        return
    H, W = first_frame.shape[:2]

    lane_json = resolve_lane_json_path(CONFIG, str(CONFIG['VIDEO']))
    lanes = _load_or_draw_lanes(first_frame, lane_json)
    if not lanes:
        st.warning("Không có lane để phân tích.")
        return

    runner = YoloRunner(weights=str(CONFIG['YOLO_WEIGHTS']),
                        device_str=str(CONFIG['DEVICE']),
                        imgsz=int(CONFIG['IMG_SIZE']),
                        per_thr=dict(CONFIG.get('CONF_PER_CLASS', {})),
                        default_thr=float(CONFIG.get('CONF', 0.35)))

    fps_ema = 0.0
    t_prev = float(cv2.getTickCount()) / cv2.getTickFrequency()

    # 👉 Dùng Streamlit container
    frame_placeholder = st.empty()
    stop_btn = st.button("⏹ Dừng phân tích")

    require_in_lane = bool(CONFIG.get('REQUIRE_IN_LANE', True))
    neutral_color = CONFIG.get('NEUTRAL_COLOR', None)
    fill_alpha = float(CONFIG['FILL_ALPHA'])
    poly_thick = int(CONFIG['POLY_THICK'])

    while cap.isOpened():
        if stop_btn:   # 👉 Nếu người dùng bấm nút dừng thì break
            break

        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        dets_xyxy, det_classes = runner.detect(frame)

        overlay = draw_lane_overlay(frame, lanes, fill_alpha=fill_alpha, poly_thick=poly_thick)
        out = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        margin = max(int(CONFIG['IN_REGION_MARGIN']), int(0.01 * W))
        for (x1, y1, x2, y2), cls in zip(dets_xyxy, det_classes):
            foot = ((x1 + x2) // 2, y2 - 3)
            in_any = False
            in_moto = False
            for lane in lanes:
                _, poly2 = poly_to_np(lane['poly'])  # type: ignore
                if inside_with_margin(poly2, foot, delta=margin):
                    in_any = True
                    in_moto = (lane['type'] == 'moto')
                    break

            if require_in_lane and not in_any:
                if isinstance(neutral_color, (tuple, list)) and len(neutral_color) == 3:
                    nc = tuple(int(c) for c in neutral_color)
                    cv2.rectangle(out, (x1, y1), (x2, y2), nc, 2, cv2.LINE_AA)
                    cv2.circle(out, foot, 4, nc, -1)
                continue

            is_bike = cls in {'motorbike', 'motorcycle', 'bicycle'}
            is_car = cls in {'car', 'truck', 'bus'}
            wrong = (is_bike and not in_moto) or (is_car and in_moto)
            draw_detection(out, x1, y1, x2, y2, cls, wrong)

        now = float(cv2.getTickCount()) / cv2.getTickFrequency()
        dt = now - t_prev
        t_prev = now
        fps_i = 1.0 / max(1e-6, dt)
        fps_ema = float(CONFIG['FPS_ALPHA']) * fps_ema + (1 - float(CONFIG['FPS_ALPHA'])) * fps_i
        put_hud(out, fps_ema, str(CONFIG['TITLE']))

        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(out_rgb, channels="RGB")

    cap.release()
    st.success("✅ Đã dừng phân tích.")
