import cv2
from ultralytics import YOLO
import numpy as np
import time
import requests
import threading

API_URL = "http://182.253.40.178:3000/api/v1/cctv"
SEND_INTERVAL = 2
# ---------------- Fungsi deteksi per CCTV ---------------- #
def run_cctv(rtsp_url: str, region_points: list, nama_cctv: str, window_name: str, model: YOLO):
    cap = cv2.VideoCapture(rtsp_url)
    assert cap.isOpened(), f"Gagal membuka RTSP stream: {nama_cctv}"
    region_np = np.array(region_points, dtype=np.int32)

    last_sent = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{nama_cctv}] Gagal membaca frame dari RTSP stream")
            break

        # Inference
        results = model(source=frame, device="intel:cpu")
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        count_inside = 0
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if cv2.pointPolygonTest(region_np, (cx, cy), False) >= 0:
                count_inside += 1
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Gambar polygon & teks
        cv2.polylines(frame, [region_np], True, (0, 255, 0), 2)
        cv2.putText(frame, f"Objek Dalam Area: {count_inside}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Kirim data tiap 2 detik
        now = time.time()
        if now - last_sent >= SEND_INTERVAL:
            payload = {"nama_cctv": nama_cctv, "objek": count_inside}
            try:
                r = requests.post(API_URL, json=payload, timeout=3)
                print(f"[{nama_cctv}] Kirim data: {payload} -> Status {r.status_code}")
            except requests.RequestException as e:
                print(f"[{nama_cctv}] Gagal kirim data: {e}")
            last_sent = now

        # Tampilkan window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# ---------------- Main ---------------- #
if __name__ == "__main__":
    # Load model sekali
    yolo_model = YOLO("yolo-orang-motor-best_openvino_model/")

    # Konfigurasi dua CCTV
    cctv_configs = [
        {
            "rtsp": "rtsp://admin:Admin123@182.253.40.178:554/Streaming/channels/301",
            "region": [(657, 336), (1600, 372), (1477, 1000), (274, 982), (121, 752)],
            "name": "CCTV Gerbang Timur",
            "window": "Deteksi Gerbang Timur"
        },
        {
            "rtsp": "rtsp://admin:Admin123@182.253.40.178:554/Streaming/channels/401",
            "region": [[167, 804], [1066, 446], [1465, 464], [1539, 390],
                       [1840, 431], [1715, 1059], [202, 1049]],
            "name": "CCTV Gerbang Barat",
            "window": "Deteksi Gerbang Barat"
        }
    ]

    threads = []
    for cctv_config in cctv_configs:
        t = threading.Thread(
            target=run_cctv,
            args=(cctv_config["rtsp"], cctv_config["region"], cctv_config["name"], cctv_config["window"], yolo_model),
            daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()