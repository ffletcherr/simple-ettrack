import random
from os.path import join
from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset

from lib.utils.cutout import Cutout
from lib.utils.utils import BBox, Center, Corner, aug_apply, center2corner


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float64)
    crop = cv2.warpAffine(
        image,
        mapping,
        (out_sz, out_sz),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=padding,
    )
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(
    image,
    bbox,
    context_amount=0.5,
    exemplar_size=127,
    instanc_size=255,
    padding=(127, 127, 127),
):
    """
    bbox: [xmin, ymin, xmax, ymax]
    """
    target_pos = [(bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


class OceanDataset(Dataset):
    def __init__(self, cfg, dataset_path: str = None, use_ettrack=False):
        super(OceanDataset, self).__init__()
        # pair information
        self.template_size = (
            cfg.ETTRACK.TRAIN.TEMPLATE_SIZE
            if use_ettrack
            else cfg.OCEAN.TRAIN.TEMPLATE_SIZE
        )
        self.search_size = (
            cfg.ETTRACK.TRAIN.SEARCH_SIZE
            if use_ettrack
            else cfg.OCEAN.TRAIN.SEARCH_SIZE
        )

        self.size = cfg.ETTRACK.TRAIN.SIZE if use_ettrack else cfg.OCEAN.TRAIN.SIZE
        self.stride = (
            cfg.ETTRACK.TRAIN.STRIDE if use_ettrack else cfg.OCEAN.TRAIN.STRIDE
        )
        self.frame_range = 60
        self.search_margin = 64

        # aug information
        self.color = cfg.OCEAN.DATASET.COLOR
        self.flip = cfg.OCEAN.DATASET.FLIP
        self.rotation = cfg.OCEAN.DATASET.ROTATION
        self.blur = cfg.OCEAN.DATASET.BLUR
        self.shift = cfg.OCEAN.DATASET.SHIFT
        self.scale = cfg.OCEAN.DATASET.SCALE
        self.gray = cfg.OCEAN.DATASET.GRAY
        self.label_smooth = cfg.OCEAN.DATASET.LABELSMOOTH
        self.mixup = cfg.OCEAN.DATASET.MIXUP
        self.cutout = cfg.OCEAN.DATASET.CUTOUT

        # aug for search image
        self.shift_s = cfg.OCEAN.DATASET.SHIFTs
        self.scale_s = cfg.OCEAN.DATASET.SCALEs

        self.grids()

        self.transform_extra = transforms.Compose(
            [
                transforms.ToPILImage(),
            ]
            + (
                [
                    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                ]
                if self.color > random.random()
                else []
            )
            + (
                [
                    transforms.RandomHorizontalFlip(),
                ]
                if self.flip > random.random()
                else []
            )
            + (
                [
                    transforms.RandomRotation(degrees=10),
                ]
                if self.rotation > random.random()
                else []
            )
            + (
                [
                    transforms.Grayscale(num_output_channels=3),
                ]
                if self.gray > random.random()
                else []
            )
            + ([Cutout(n_holes=1, length=16)] if self.cutout > random.random() else [])
        )

        self.dataset_path = Path(dataset_path)
        self.tracks_path = sorted(
            [folder_path for folder_path in self.dataset_path.glob("*")]
        )
        self.get_track_infos()
        print(cfg)

    def __len__(self):
        return len(self.tracks_info)

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        template, search = self._get_pairs(index)

        template_image = cv2.imread(template[0].as_posix())
        search_image = cv2.imread(search[0].as_posix())

        # change bboxes format and pick the first one
        template_target_bbox = self.yolo2ocean(template[1], template_image)
        search_target_bbox = self.yolo2ocean(search[1], search_image)
        _, template_image = crop_like_SiamFC(
            template_image,
            bbox=template_target_bbox,
            exemplar_size=self.template_size,
            instanc_size=self.search_size,
        )
        _, search_image = crop_like_SiamFC(
            search_image,
            bbox=search_target_bbox,
            exemplar_size=self.template_size,
            instanc_size=self.search_size + self.search_margin,
        )
        template_box = self._toBBox(template_image, template_target_bbox)
        search_box = self._toBBox(search_image, search_target_bbox)

        template, _, _ = self._augmentation(
            template_image, template_box, self.template_size
        )
        search, bbox, dag_param = self._augmentation(
            search_image, search_box, self.search_size, search=True
        )

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)

        out_label = self._dynamic_label([self.size, self.size], dag_param.shift)

        reg_label, reg_weight = self.reg_label(bbox)
        template, search = map(
            lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search]
        )

        return (
            template,
            search,
            out_label,
            reg_label,
            reg_weight,
            np.array(bbox, np.float32),
        )  # self.label 15*15/17*17

    # ------------------------------------
    # function groups for selecting pairs
    # ------------------------------------
    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.size

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(
            np.arange(0, sz) - np.floor(float(sz_x)),
            np.arange(0, sz) - np.floor(float(sz_y)),
        )

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

    def reg_label(self, bbox):
        """
        generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1  # [17, 17]
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)  # [17, 17, 4]
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def get_track_infos(self):
        self.tracks_info = {}
        for track_idx, track_folder in enumerate(self.tracks_path):
            txt_path_list = sorted(list(track_folder.glob("*.txt")))
            frames_path_dict = {
                idx: {"frame": txt_path.with_suffix(".jpg"), "label": txt_path}
                for idx, txt_path in enumerate(txt_path_list)
            }
            self.tracks_info[track_idx] = frames_path_dict

    def _get_image_anno(
        self, track_dict: dict[int, dict[str, Path]], frame_idx: int
    ) -> tuple[Path, list[float]]:
        """
        get image and annotation
        """
        frame_path = track_dict[frame_idx]["frame"]
        label_path = track_dict[frame_idx]["label"]
        image_anno = [eval(line) for line in label_path.read_text().split("\n")]
        assert len(image_anno) > 0, "image_anno is empty"
        image_anno = image_anno[0]
        return frame_path, image_anno

    def _get_pairs(self, index):
        """
        get training pairs
        """
        track_dict = self.tracks_info[index]
        frames = list(range(len(track_dict)))
        template_frame = random.randint(0, len(frames) - 1)  # 2

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = int(frames[template_frame])
        search_frame = int(random.choice(search_range))  # search_range[2]

        return self._get_image_anno(track_dict, template_frame), self._get_image_anno(
            track_dict, search_frame
        )

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def yolo2ocean(self, bbox, image):
        """
        normalized yolo format bounding box: [class, x, y, w, h]
        """
        imh, imw = image.shape[:2]
        bc, bx, by, bw, bh = bbox
        x = bx * imw
        y = by * imh
        w = bw * imw
        h = bh * imh
        ocean_bbox = list(map(int, [x, y, x + w, y + h]))
        return ocean_bbox

    def _toBBox(self, image, shape):
        """
        create a boundig box for cropped image (search image)
        from bounding box that is in main image coordinates system
        shape: [x1, x2, y1, y2]
        """
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(
            image,
            mapping,
            (out_sz, out_sz),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=padding,
        )
        return crop

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """
        draw_image = np.array(image.copy())
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(
            draw_image,
            (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
            3,
            (0, 0, 255),
        )
        cv2.putText(
            draw_image,
            "[x: {}, y: {}]".format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
            (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )
        cv2.imwrite(name, draw_image)

    def _draw_reg(self, image, grid_x, grid_y, reg_label, reg_weight, save_path, index):
        """
        visiualization
        reg_label: [l, t, r, b]
        """
        draw_image = image.copy()
        # count = 0
        save_name = join(save_path, "{:06d}.jpg".format(index))
        h, w = reg_weight.shape
        for i in range(h):
            for j in range(w):
                if not reg_weight[i, j] > 0:
                    continue
                else:
                    x1 = int(grid_x[i, j] - reg_label[i, j, 0])
                    y1 = int(grid_y[i, j] - reg_label[i, j, 1])
                    x2 = int(grid_x[i, j] + reg_label[i, j, 2])
                    y2 = int(grid_y[i, j] + reg_label[i, j, 3])

                    draw_image = cv2.rectangle(
                        draw_image, (x1, y1), (x2, y2), (0, 255, 0)
                    )

        cv2.imwrite(save_name, draw_image)

    def _mixupRandom(self):
        """
        gaussian random -- 0.3~0.7
        """
        return random.random() * 0.4 + 0.3

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size, search=False):
        """
        data augmentation for input pairs (modified from SiamRPN.)
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if search:
            param.shift = (
                self._posNegRandom() * self.shift_s,
                self._posNegRandom() * self.shift_s,
            )  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_s),
                (1.0 + self._posNegRandom() * self.scale_s),
            )  # scale change
        else:
            param.shift = (
                self._posNegRandom() * self.shift,
                self._posNegRandom() * self.shift,
            )  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale),
                (1.0 + self._posNegRandom() * self.scale),
            )  # scale change
        crop_bbox, real_param = aug_apply(Corner(*crop_bbox), param, shape)
        real_param = edict(real_param)
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(
            bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y
        )

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)  # other data augmentation
        return image, bbox, real_param

    def _mixupShift(self, image, size):
        """
        random shift mixed-up image
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * 64, self._posNegRandom() * 64)  # shift
        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image

    # ------------------------------------
    # function for creating training label
    # ------------------------------------
    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        # assert fixedLabelSize[0] % 2 == 1

        d_label = self._create_dynamic_logisticloss_label(
            fixedLabelSize, c_shift, rPos, rNeg
        )

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        sz_x = sz // 2 + int(-c_shift[0] / self.stride)  # 8 is strides
        sz_y = sz // 2 + int(-c_shift[1] / self.stride)

        x, y = np.meshgrid(
            np.arange(0, sz) - np.floor(float(sz_x)),
            np.arange(0, sz) - np.floor(float(sz_y)),
        )

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(
            dist_to_center <= rPos,
            np.ones_like(y),
            np.where(dist_to_center < rNeg, 0.5 * np.ones_like(y), np.zeros_like(y)),
        )
        return label


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    from parameters import parameters
    from lib.core.config_ocean import config
    import torch

    dataset_path = Path(os.environ["dataset_path"])
    dataset = OceanDataset(
        cfg=config, dataset_path=dataset_path / "tracks", use_ettrack=True
    )

    (
        template,
        search,
        out_label,
        reg_label,
        reg_weight,
        bbox,
    ) = dataset[222]
    template, search = map(
        lambda x: np.transpose(x, (1, 2, 0)).astype(np.uint8), [template, search]
    )
    reg_weight = cv2.cvtColor(reg_weight.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    reg_weight = cv2.resize(reg_weight, (search.shape[1], search.shape[0]))
    out_label = cv2.cvtColor(out_label.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    out_label = cv2.resize(out_label, (search.shape[1], search.shape[0]))
    x1, y1, x2, y2 = map(int, bbox)
    search = cv2.rectangle(search * reg_weight, (x1, y1), (x2, y2), (200, 100, 150))
    cv2.imshow("search", search)
    cv2.imshow("out_label", out_label)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # params = parameters()
    # model = params.net
    # model.initialize(params.model_name, checkpoint_epoch=params.checkpoint_epoch)

    # template, search = map(
    #     lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search]
    # )

    # template = torch.tensor(template).unsqueeze(0)
    # search = torch.tensor(search).unsqueeze(0)
    # out_label = torch.tensor(out_label).unsqueeze(0)
    # reg_label = torch.tensor(reg_label).unsqueeze(0)
    # reg_weight = torch.tensor(reg_weight).unsqueeze(0)
    # out = model(
    #     template, search, out_label, reg_target=reg_label, reg_weight=reg_weight
    # )
