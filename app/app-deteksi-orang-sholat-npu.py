import cv2
from ultralytics import YOLO
import numpy as np

def main():
    # RTSP URL (dengan user & password)
    rtsp_url = "rtsp://admin:Admin123@192.168.18.186:554/Streaming/channels/1301"

    # Buka stream
    cap = cv2.VideoCapture(rtsp_url)
    assert cap.isOpened(), "Gagal membuka RTSP stream"

    # Definisikan region polygon
    region_points = [(70, 468), (803, 309), (1202, 668), (1054, 1156), (787, 1301), (118, 1288)]
    region_np = np.array(region_points, dtype=np.int32)

    # Load model YOLO
    ov_model = YOLO("orang-sholat-v2-best_openvino_model/")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari RTSP stream")
            break

        # Inference hanya person
        results = ov_model(source=frame, device="intel:npu")

        # Ambil hasil deteksi
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        count_inside = 0
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # cek apakah centroid ada di dalam polygon
            inside = cv2.pointPolygonTest(region_np, (cx, cy), False)
            if inside >= 0:
                count_inside += 1
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # gambar polygon region
        cv2.polylines(frame, [region_np], isClosed=True, color=(0, 0, 255), thickness=2)

        # tampilkan jumlah orang di dalam region
        cv2.putText(frame, f"Jumlah Jamaah: {count_inside}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # tampilkan hasil
        cv2.namedWindow("Deteksi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteksi", 800, 600)
        cv2.imshow("Deteksi", frame)

        # tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Deteksi", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
