from collections import defaultdict
import cv2

from ultralytics import YOLO
from pathlib import Path
import numpy as np

runs_dir = Path("runs/detect")

latest_dir_with_model = 0

# find latest dir with model weights
for dir in runs_dir.iterdir():
    weights_dir = dir / "weights"
    if weights_dir.exists():
        if (weights_dir / "best.pt").exists() and (weights_dir / "last.pt").exists():
            train_num = dir.name.split("train")[-1]
            train_num = int(train_num)
            if train_num > latest_dir_with_model:
                latest_dir_with_model = train_num

if latest_dir_with_model <= 0:
    print("Could not find model")
    exit(0)

model = YOLO(runs_dir / f"train{latest_dir_with_model}" / "weights" / "best.pt").to(
    "cuda"
)


cap = cv2.VideoCapture(
    # "./../data_processing/sample_every_n_frames/videos_old/C-20241019-115254-1729363974656-7.mp4"
    # "./../videos/C-20241022-133240/C-20241022-133240-1729629160225-7.mp4"
    "./../../../../Downloads/A-20241022-154226-1729636946974-7.mp4"
    # "./../../../../Downloads/B-20241022-154157-1729636917745-7.mp4"
    # "./../../../../Downloads/C-20241022-154240-1729636960091-7.mp4"
)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot(line_width=1, conf=False, labels=False)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=2,
            )

        cv2.imshow("Bee Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cap.destroyAllWindows()
