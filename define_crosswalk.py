import cv2
import json
import numpy as np
#MISSING VID ??# - Moss
#cap = cv2.VideoCapture("video/01.mp4")
cap = cv2.VideoCapture("TestVid02.mp4")
#cap = cv2.VideoCapture("rtsp://admin:1212312121@192.168.1.190:10554/udp/av0_0")

zone_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append([x, y])

ret, frame = cap.read()
if not ret:
    print("cant open cam")
    cap.release()
    exit()

clone = frame.copy()
cv2.namedWindow("Define Crosswalk Zone")
cv2.setMouseCallback("Define Crosswalk Zone", click_event)

while True:
    temp = clone.copy()
    if len(zone_points) > 0:
        for i, point in enumerate(zone_points):
            cv2.circle(temp, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(temp, str(i+1), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if len(zone_points) >= 2:
            cv2.polylines(temp, [np.array(zone_points)], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.imshow("Define Crosswalk Zone", temp)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        zone_points.clear()
        print("new start")
    elif key == ord("s") and len(zone_points) >= 3:
        with open("crosswalk_zone.json", "w") as f:
            json.dump(zone_points, f)
        print("save crosswalk_zone.json แล้ว")
        break

cv2.destroyAllWindows()
cap.release()
