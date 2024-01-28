import os
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()


def jeval(line):
    if ";" in line:
        boxes = [eval(b) for b in line.split(";")[:-1]]
    else:
        boxes = []
    return boxes


dataset_path = Path(os.environ["dataset_path"])
videos_list = list(dataset_path.glob("*/*/*.mp4"))

for video_path in videos_list:
    label_path = video_path.with_suffix(".txt")
    labels = [jeval(line) for line in label_path.read_text().split("\n")[:-1]]
    cap = cv2.VideoCapture(video_path.as_posix())
    fn = 0
    tn = 0
    new_track_name = None
    video_name = video_path.stem
    track_started = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        bboxes = labels[fn]
        if bboxes:
            if not track_started:
                track_started = True
                new_track_name = f"{video_name}-{tn:03d}"
            for bbox in bboxes:
                c, x, y, w, h = map(int, bbox[:5])
        else:
            track_started = False
        fn += 1
