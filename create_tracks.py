import os
from pathlib import Path

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

for vide_path in videos_list:
    label_path = vide_path.with_suffix(".txt")
    labels = [jeval(line) for line in label_path.read_text().split("\n")[:-1]]
