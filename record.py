import cv2, time

SECONDS, FPS, SHOT_EVERY = 10, 15, 2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
assert cap.isOpened(), "카메라를 열 수 없음 (/dev/video0)"
for _ in range(10):
    cap.read()

out, n, shot, next_shot = None, 0, 1, 0.0
print("녹화 시작! 카메라를 천천히 돌려 주세요")
t0 = time.time()
while (el := time.time() - t0) < SECONDS:
    ok, frame = cap.read()
    if not ok:
        continue
    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter("media/clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    while n < el * FPS:
        out.write(frame)
        n += 1
    if el >= next_shot:
        cv2.imwrite(f"media/frame_{shot:02d}.jpg", frame)
        print(f"  {int(el)}초: frame_{shot:02d}.jpg 저장")
        shot += 1
        next_shot += SHOT_EVERY
out.release()
cap.release()
print(f"완료: clip.mp4 ({n}프레임, {SECONDS}초), 사진 {shot - 1}장")
