from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv
import time

# Inisialisasi model YOLOv8
model = YOLO('yolov8n.pt')

# Inisialisasi Deep SORT
tracker = DeepSort(max_age=30, n_init=3)

# Video input
video_path = 'videos/test_video.mp4'
cap = cv2.VideoCapture(video_path)

# File log
log_file = open('movement_log.csv', mode='w', newline='')
writer = csv.writer(log_file)
writer.writerow(['frame', 'timestamp', 'track_id', 'x_center', 'y_center', 'confidence'])

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = round(time.time() - start_time, 2)

    # Deteksi manusia saja
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == 'person' and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Jalankan Deep SORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        x_center = int((l + w) / 2)
        y_center = int((t + h) / 2)
        conf = round(track.det_conf, 3) if track.det_conf else 0

        writer.writerow([frame_count, timestamp, track_id, x_center, y_center, conf])

        cv2.rectangle(frame, (int(l), int(t)), (int(w), int(h)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
print("Tracking selesai. Log disimpan di movement_log.csv")
