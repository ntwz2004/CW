from ultralytics import YOLO
import cv2
import numpy as np
import json
import time

model = YOLO("yolo11n.pt")

# โหลด class ที่ต้องการ
target_classes = ['car', 'motorcycle', 'truck', 'bus','traffic light']
class_names = model.names
target_class_ids = [cls_id for cls_id, name in class_names.items() if name in target_classes]

# โหลดเขตทางม้าลาย
with open("crosswalk_zone.json", "r") as f:
    crosswalk_zone = json.load(f)
crosswalk_zone_np = np.array(crosswalk_zone, np.int32)

# กล้อง
cap = cv2.VideoCapture("testTraf.mp4") 
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def is_inside_crosswalk(center, polygon_points):
    return cv2.pointPolygonTest(np.array(polygon_points), center, False) >= 0

frame_count = 0
SKIP_FRAME = 1  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1920, 1080))

    frame_count += 1
    start = time.time()
    results = model(frame, verbose=False)[0]
    end = time.time()

    # ทางม้าลาย
    cv2.polylines(frame, [crosswalk_zone_np], isClosed=True, color=(0, 255, 255), thickness=2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in target_class_ids:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(box.conf[0])
            label = class_names[cls_id]

            # วาดกรอบวัตถุ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ตรวจจับว่าอยู่ในทางม้าลายหรือไม่
            if is_inside_crosswalk((cx, cy), crosswalk_zone):
                cv2.putText(frame, "In crosswalk", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # 🔴🟡🟢 ตรวจจับสีของ traffic light
            if label == 'traffic light':
                light_roi = frame[y1:y2, x1:x2]
                if light_roi.size == 0:
                    continue

            

                # แสดงผล
                cv2.putText(frame, f"{}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    fps = 1 / (end - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Pi5", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    