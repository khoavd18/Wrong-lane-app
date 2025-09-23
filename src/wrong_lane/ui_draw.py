# src/wrong_lane/ui_draw.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Optional
import cv2
from .lanes import save_lanes_json, load_lanes_json
from .overlay import draw_lane_overlay
from .config import CONFIG   

def ask_lane_counts(m: int, o: int) -> Tuple[int, int]:
    return max(0, m), max(0, o)

def draw_lanes_ui(first_frame, json_path: str, fill_alpha: float, poly_thick: int) -> Optional[List[Dict]]:
    H, W = first_frame.shape[:2]
    win = 'Wrong-lane (Draw Lanes)'
    cv2.namedWindow(win)

    cnt_moto, cnt_other = ask_lane_counts(CONFIG.get('LANE_MOTO', 1), CONFIG.get('LANE_OTHER', 2))

    target_seq: List[str] = (['moto'] * cnt_moto) + (['other'] * cnt_other)

    lanes: List[Dict] = []
    current_points: List[Tuple[int,int]] = []
    idx_lane = 0

    # callback chuột
    def on_mouse(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 4:
            current_points.append((x, y))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        display = first_frame.copy()
        cv2.putText(display,
                    'DRAW: click 4 pts -> SPACE confirm | U=undo | R=reset | S=save | L=load | ESC=quit',
                    (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        next_txt = target_seq[idx_lane] if idx_lane < len(target_seq) else 'DONE'
        cv2.putText(display, f'Next lane type: {next_txt}',
                    (15,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

        # vẽ overlay lane đã có
        if lanes:
            display = draw_lane_overlay(display, lanes, fill_alpha=fill_alpha, poly_thick=poly_thick)

        # vẽ các điểm đang chọn
        for i, p in enumerate(current_points):
            cv2.circle(display, p, 5, (0,255,255), -1)
            if i > 0:
                cv2.line(display, current_points[i-1], p, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        if key == ord('u'):  # undo
            if current_points: current_points.pop()
        elif key == ord('r'):  # reset
            lanes.clear(); current_points.clear(); idx_lane = 0
        elif key == ord('s'):  # save tạm
            save_lanes_json(json_path, lanes, meta={'size':[W,H]})
            print(f'Da luu {json_path}')
        elif key == ord('l'):  # load lại
            data = load_lanes_json(json_path)
            if data and 'lanes' in data:
                lanes = data['lanes']  # type: ignore
                current_points.clear()
                idx_lane = len(lanes)
                print(f'Da load {json_path} ({len(lanes)} lanes)')
                if idx_lane >= len(target_seq): break
            else:
                print('Khong tim thay/khong hop le JSON')
        elif key == 32:  # SPACE confirm
            if idx_lane < len(target_seq) and len(current_points) == 4:
                lanes.append({'type': target_seq[idx_lane], 'poly': current_points.copy()})
                current_points.clear()
                idx_lane += 1
                if idx_lane >= len(target_seq): break

        if idx_lane >= len(target_seq):
            break
        # kiểm tra nút X (close window)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    # lưu kết quả cuối cùng
    save_lanes_json(json_path, lanes, meta={'size':[W,H]})
    print(f'[OK] Ve xong {len(lanes)} lan, saved -> {json_path}')

    cv2.destroyAllWindows()
    return lanes
