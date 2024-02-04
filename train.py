from pathlib import Path

import cv2
import numpy as np

from data_utils import OceanDataset
from lib.core.config_ocean import config

if __name__ == "__main__":
    import os

    dataset_path = Path(os.environ["dataset_path"])
    dataset = OceanDataset(
        cfg=config,
        dataset_path=dataset_path / "tracks",
    )

    template, search, out_label, reg_label, reg_weight, bbox = dataset[22]
    out_label = (cv2.resize(out_label, search.shape[:2]) * 255).astype(np.uint8)
    cv2.imshow("template", template)
    cv2.imshow("search", search)
    cv2.imshow("out_label", out_label)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
