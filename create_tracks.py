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

max_track_len = int(os.environ.get("max_track_len", 512))


def dump_track(track_frames, track_labels, dataset_path: Path, new_track_name: str):
    track_path = dataset_path / "tracks" / new_track_name
    track_path.mkdir(exist_ok=True, parents=True)
    for idx, (t_frame, t_label) in enumerate(zip(track_frames, track_labels)):
        cv2.imwrite(f"{track_path}/{idx:04d}.jpg", t_frame)
        file = track_path / f"{idx:04d}.txt"
        file.write_text(
            "\n".join([f"{c},{x},{y},{w},{h}" for c, x, y, w, h in t_label])
        )
    track_frames = []
    track_labels = []
    return track_frames, track_labels


for video_path in videos_list:
    label_path = video_path.with_suffix(".txt")
    labels = [jeval(line) for line in label_path.read_text().split("\n")[:-1]]
    cap = cv2.VideoCapture(video_path.as_posix())
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fn = 0
    tn = 0
    new_track_name = ""
    video_name = video_path.stem
    track_started = False
    track_frames = []
    track_labels = []
    print("video_path:", video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        bboxes = labels[fn]
        if bboxes:
            if not track_started:
                track_started = True
                new_track_name = f"{video_name}-{tn:03d}"
                tn += 1
            bbox_list = []
            for bbox in bboxes:
                c, x, y, w, h = map(int, bbox[:5])
                x = x / width
                y = y / height
                w = w / width
                h = h / height
                bbox_list.append([c, x, y, w, h])
            track_frames.append(frame)
            track_labels.append(bbox_list)
            if len(track_frames) >= max_track_len:
                track_started = False
                track_frames, track_labels = dump_track(
                    track_frames, track_labels, dataset_path, new_track_name
                )
        else:
            if track_started:
                track_started = False
                track_frames, track_labels = dump_track(
                    track_frames, track_labels, dataset_path, new_track_name
                )
        fn += 1
    if track_started:
        track_started = False
        track_frames, track_labels = dump_track(
            track_frames, track_labels, dataset_path, new_track_name
        )
