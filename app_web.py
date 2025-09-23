import streamlit as st
import os
import logging
from src.wrong_lane.app import main as app_main
from src.wrong_lane.config import CONFIG

# --- CẤU HÌNH LOG ---
LOG_FILE = "app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log_event(msg: str):
    logging.info(msg)


st.set_page_config(page_title="🚦 NHÓM 13", layout="wide")

# BANNER CHÍNH
st.markdown(
    """
    <div style="text-align:center; padding:15px; background:#1E88E5; color:white; border-radius:12px; margin-bottom:20px">
        <h1>🚗🏍 NHÓM 13</h1>
        <p>Ứng dụng phát hiện đi sai làn bằng YOLO + OpenCV</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- BỐ CỤC UPLOAD ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📹 Tải Video")
    video_file = st.file_uploader("Chọn VIDEO (.mp4/.avi)", type=["mp4", "avi"])

with col2:
    st.subheader("📄 Tải File JSON (tùy chọn)")
    json_file = st.file_uploader("Chọn file JSON lane", type=["json"])

st.subheader("⚙️ Cấu hình chế độ")
mode_display = st.radio("Chọn chế độ khởi động:", ["Vẽ config mới", "Auto-load config"], horizontal=True)

mode = "draw" if mode_display == "Vẽ config mới" else "json_custom"

if mode == "draw":
    with st.expander("✏️ Tùy chỉnh số lượng lane"):
        m = st.number_input("Nhập số lane XE MÁY:", min_value=0, max_value=5, value=1, step=1)
        o = st.number_input("Nhập số lane XE KHÁC (ô tô, bus, truck...):", min_value=0, max_value=5, value=2, step=1)
        CONFIG['LANE_MOTO'] = m
        CONFIG['LANE_OTHER'] = o
        log_event(f"Config lanes: moto={m}, other={o}")

st.markdown("---")

if video_file is not None:
    log_event(f"User uploaded video: {video_file.name}")

    if st.button("🚀 Bắt đầu phân tích"):
        if mode == "json_custom" and json_file is None:
            st.error("❌ Bạn cần tải lên file JSON khi chọn chế độ Auto-load config.")
            log_event("Error: JSON file required but not uploaded")
        else:
            # LƯU TẠM VIDEO
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            os.environ["VIDEO"] = video_path
            log_event(f"Saved temp video -> {video_path}")

            # LƯU TẠM JSON (NẾU CÓ)
            if json_file is not None:
                json_path = "temp_lanes.json"
                with open(json_path, "wb") as f:
                    f.write(json_file.read())
                CONFIG['LANE_JSON_PATH'] = json_path
                log_event(f"Loaded custom lane JSON -> {json_path}")

            CONFIG['START_MODE'] = mode
            log_event(f"Start mode = {mode}")

            from src.wrong_lane.app import main_streamlit

            with st.spinner("🔄 Đang chạy mô hình YOLO + OpenCV... Vui lòng chờ"):
                log_event("Start running YOLO + OpenCV (main_streamlit)")
                main_streamlit()
            st.success("✅ Hoàn tất! Video đã được phân tích và hiển thị ở trên.")
            log_event("Analysis finished successfully")

else:
    st.info("⬆️ Hãy tải lên video để bắt đầu.")

# --- HIỂN THỊ LOG TRONG GIAO DIỆN ---
st.markdown("---")
if st.checkbox("📑 Xem log"):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            st.text(f.read())
