import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("无法打开摄像头：请确认已关闭 VideoCap，并尝试摄像头编号 2")
    raise SystemExit

print("摄像头已打开，按 Q 退出预览")
while True:
    ok, frame = cap.read()
    if not ok:
        print("读取摄像头画面失败")
        break
    cv2.imshow("Camera Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
