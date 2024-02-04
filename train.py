import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_utils import OceanDataset
from et_track import ET_Tracker
from lib.core.config_ocean import config
from lib.core.function import ocean_train
from lib.utils.utils import build_lr_scheduler, create_logger
from parameters import parameters


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("trainable params:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, "no trainable parameters"

    return trainable_params


if __name__ == "__main__":

    gpus = [0]
    gpu_num = len(gpus)
    device = torch.device(
        "cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu"
    )
    logger, _, tb_log_dir = create_logger(config, "OCEAN", "train")
    dataset_path = Path(os.environ["dataset_path"])

    model = ET_Tracker(
        search_size=256,
        template_size=128,
        stride=16,
        e_exemplars=4,
        sm_normalization=True,
        temperature=2,
        dropout=False,
    )
    params = parameters()
    model = params.net
    model.initialize(params.model_name, checkpoint_epoch=params.checkpoint_epoch)
    trainable_params = check_trainable(model, logger)
    optimizer = torch.optim.SGD(
        trainable_params,
        config.OCEAN.TRAIN.LR,
        momentum=config.OCEAN.TRAIN.MOMENTUM,
        weight_decay=config.OCEAN.TRAIN.WEIGHT_DECAY,
    )

    lr_scheduler = build_lr_scheduler(
        optimizer, config, epochs=config.OCEAN.TRAIN.END_EPOCH, modelFLAG="OCEAN"
    )
    lr_scheduler.step(config.OCEAN.TRAIN.START_EPOCH)
    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
    }
    for epoch in range(config.OCEAN.TRAIN.START_EPOCH, config.OCEAN.TRAIN.END_EPOCH):
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
        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()
        model, writer_dict = ocean_train(
            data_loader,
            model,
            optimizer,
            epoch + 1,
            curLR,
            config,
            writer_dict,
            logger,
            device=device,
        )
