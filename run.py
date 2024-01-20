from pathlib import Path
import os
import cv2
from dotenv import load_dotenv

from et_tracker import TransconverTracker
from parameters import parameters

load_dotenv()

tracker = TransconverTracker(parameters())
tracker.initialize_features()

dataset_path = Path(os.environ["dataset_path"])
folder_path = dataset_path / os.environ["folder_path"]
video_name = os.environ["video_name"]

video_path = (folder_path / video_name).with_suffix(".mp4")
label_path = (folder_path / video_name).with_suffix(".txt")
cap = cv2.VideoCapture(video_path.as_posix())
frames_count = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
labels = label_path.read_text()
labels = labels.split("\n")[:-1]


def jeval(line):
    if ";" in line:
        boxes = [eval(b) for b in line.split(";")[:-1]]
    else:
        boxes = []
    return boxes


labels = [jeval(line) for line in labels]
assert len(labels) == frames_count

fn = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    bboxes = labels[fn]
    for bbox in bboxes:
        c, x, y, w, h = map(int, bbox[:5])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 255, 12))
    cv2.imshow("f", frame)
    k = cv2.waitKey()
    if k == ord("q"):
        break
    fn += 1

cap.release()
cv2.destroyAllWindows()
