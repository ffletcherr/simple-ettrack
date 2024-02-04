import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from data_utils import OceanDataset
from lib.core.config_ocean import config
from lib.core.function import ocean_train

if __name__ == "__main__":

    gpu_num = 1
    dataset_path = Path(os.environ["dataset_path"])
    dataset = OceanDataset(
        cfg=config,
        dataset_path=dataset_path / "tracks",
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.OCEAN.TRAIN.BATCH * gpu_num,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=None,
        drop_last=True,
    )

    batch = next(iter(data_loader))

    (
        template,
        search,
        out_label,
        reg_label,
        reg_weight,
        bbox,
    ) = batch
