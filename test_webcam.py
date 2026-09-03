import os
import time

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best.pt"

# Iriun Webcam
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_MSMF

# YOLO
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640

# Cửa sổ hiển thị
WINDOW_NAME = "AI Nhan Dien Rac"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700


# ============================================================
# MAIN
# ============================================================

def run_webcam():

    # --------------------------------------------------------
    # Kiểm tra model
    # --------------------------------------------------------

    if not os.path.isfile(MODEL_PATH):
        print(f"Khong tim thay model: {MODEL_PATH}")
        return

    # --------------------------------------------------------
    # Chọn thiết bị xử lý
    # --------------------------------------------------------

    device = 0 if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:
        model = YOLO(MODEL_PATH)

        if torch.cuda.is_available():
            model.to("cuda")

    except Exception as e:
        print(f"Loi khi load model: {e}")
        return

    # --------------------------------------------------------
    # Mở Iriun Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        CAMERA_BACKEND
    )

    if not cap.isOpened():
        print("Khong the mo Iriun Webcam.")
        cap.release()
        return

    # --------------------------------------------------------
    # Thiết lập camera
    # --------------------------------------------------------

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --------------------------------------------------------
    # Tạo cửa sổ
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    prev_time = time.perf_counter()
    fps = 0.0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # ----------------------------------------------------
        # YOLO inference
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )[0]

        # ----------------------------------------------------
        # Vẽ kết quả YOLO
        # ----------------------------------------------------

        annotated = results.plot()

        # ----------------------------------------------------
        # Tính FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()
        elapsed = current_time - prev_time
        prev_time = current_time

        current_fps = 1.0 / max(elapsed, 1e-6)

        # Làm FPS ổn định hơn
        fps = fps * 0.9 + current_fps * 0.1

        # ----------------------------------------------------
        # Hiển thị FPS
        # ----------------------------------------------------

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Hiển thị
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            annotated
        )

        # ----------------------------------------------------
        # Q / ESC để thoát
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        # Nếu đóng cửa sổ bằng X
        if cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_webcam()