import time
from collections import OrderedDict

import cv2
import numpy as np
import torch


def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)
def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img

def cxy_wh_2_rect(pos, sz):
    return [float(max(float(0), pos[0]-sz[0]/2)), float(max(float(0), pos[1]-sz[1]/2)), float(sz[0]), float(sz[1])]  # 0-index


def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode="torch"):
    """
    SiamFC type cropping
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round((pos[0] - c).item())
    context_xmax = context_xmin + sz - 1
    context_ymin = round((pos[1] - c).item())
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0.0, -context_xmin))
    top_pad = int(max(0.0, -context_ymin))
    right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros(
            (r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8
        )
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad :, :] = avg_chans
        im_patch_original = te_im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[
            int(context_ymin) : int(context_ymax + 1),
            int(context_xmin) : int(context_xmax + 1),
            :,
        ]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    crop_info["crop_cords"] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info["empty_mask"] = tete_im
    crop_info["pad_info"] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params
        self.visdom = None

    def predicts_segmentation_mask(self):
        return False

    def initialize(self, image, info: dict) -> dict:
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image, info: dict = None) -> dict:
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), "Tracking", 1, "Tracking")
        else:
            self.visdom.register((image, *box, segmentation), "Tracking", 1, "Tracking")


class TransconverTracker(BaseTracker):
    def initialize_features(self):
        if not getattr(self, "features_initialized", False):
            checkpoint_epoch = self.params.get("checkpoint_epoch", None)
            self.params.net.initialize(
                self.params.model_name, checkpoint_epoch=checkpoint_epoch
            )

        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        """initialize the model"""

        state_dict = dict()

        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has("device"):
            self.params.device = "cuda" if self.params.use_gpu else "cpu"

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Initialize network
        # verify that the model is correctly initialized:
        self.initialize_features()

        # The Baseline network
        self.net = self.params.net
        self.net.eval()
        self.net.to(self.params.device)

        self.weight_style = self.params.get("weight_style", "regular")
        print(f"tracker weight style: {self.weight_style}")

        # Time initialization
        tic = time.time()

        # Get target position and size
        state = torch.tensor(info["init_bbox"])  # x,y,w,h
        cx, cy, w, h = get_axis_aligned_bbox(state)
        self.target_pos = np.array([cx, cy])
        self.target_sz = np.array([w, h])
        # self.target_pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        # self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get("object_ids", [None])[0]
        self.id_str = "" if self.object_id is None else " {}".format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([image.shape[0], image.shape[1]])
        sz = self.params.image_sample_size  # search size (256, 256)
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz
        self.stride = self.params.stride

        # LightTrack specific parameters
        p = Config(stride=self.stride, even=self.params.even)

        state_dict["im_h"] = image.shape[0]
        state_dict["im_w"] = image.shape[1]

        if (
            (self.target_sz[0] * self.target_sz[1])
            / float(state_dict["im_h"] * state_dict["im_w"])
        ) < 0.004:
            p.instance_size = (
                self.params.big_sz
            )  # cfg_benchmark['big_sz']  # -> p.instance_size = 288
            p.renew()
        else:
            p.instance_size = (
                self.params.small_sz
            )  # cfg_benchmark['small_sz'] # -> p.instance_size = 256
            p.renew()

        # compute grids
        self.grids(p)

        wc_z = self.target_sz[0] + p.context_amount * sum(self.target_sz)
        hc_z = self.target_sz[1] + p.context_amount * sum(self.target_sz)
        s_z = round(np.sqrt(wc_z * hc_z).item())

        avg_chans = np.mean(image, axis=(0, 1))
        z_crop, _ = get_subwindow_tracking(
            image, self.target_pos, p.exemplar_size, s_z, avg_chans
        )
        z_crop = self.normalize(z_crop)
        z = z_crop.unsqueeze(0)
        self.net.template(z.to(self.params.device))

        if p.windowing == "cosine":
            window = np.outer(
                np.hanning(p.score_size), np.hanning(p.score_size)
            )  # [17,17]
        elif p.windowing == "uniform":
            window = np.ones(int(p.score_size), int(p.score_size))
        else:
            raise ValueError("Unsupported window type")

        state_dict["p"] = p
        state_dict["avg_chans"] = avg_chans
        state_dict["window"] = window
        state_dict["target_pos"] = self.target_pos
        state_dict["target_sz"] = self.target_sz
        state_dict["time"] = time.time() - tic

        return state_dict

    def update(
        self,
        x_crops,
        target_pos,
        target_sz,
        window,
        scale_z,
        p,
        debug=False,
        writer=None,
    ):
        # cls_score, bbox_pred = self.net.track(x_crops)
        # cls_score, bbox_pred = self.net.track(x_crops.cuda())
        cls_score, bbox_pred = self.net.track(x_crops.to(self.params.device))
        cls_score = torch.sigmoid(cls_score).squeeze().cpu().data.numpy()

        # bbox to real predict
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # size penalty
        s_c = self.change(
            self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz))
        )  # scale penalty
        r_c = self.change(
            (target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1))
        )  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # get max
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = (
            diff_xs / scale_z,
            diff_ys / scale_z,
            pred_w / scale_z,
            pred_h / scale_z,
        )

        target_sz = target_sz / scale_z

        # size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        if debug:
            return target_pos, target_sz, cls_score[r_max, c_max], cls_score
        else:
            return target_pos, target_sz, cls_score[r_max, c_max]

    def track(self, im, state_input, writer=None):
        self.frame_num += 1

        state = state_input["previous_output"]
        p = state["p"]
        avg_chans = state["avg_chans"]
        window = state["window"]
        target_pos = state["target_pos"]
        target_sz = state["target_sz"]

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)

        scale_z = p.exemplar_size / s_z
        d_search = (
            p.instance_size - p.exemplar_size
        ) / 2  # slightly different from rpn++
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        x_crop, _ = get_subwindow_tracking(
            im, target_pos, p.instance_size, python2round(s_x.item()), avg_chans
        )
        state["x_crop"] = x_crop.clone()  # torch float tensor, (3,H,W)
        x_crop = self.normalize(x_crop)
        x_crop = x_crop.unsqueeze(0)
        debug = True

        if debug:
            target_pos, target_sz, _, cls_score = self.update(
                x_crop,
                target_pos,
                target_sz * scale_z,
                window,
                scale_z,
                p,
                debug=debug,
                writer=writer,
            )
            state["cls_score"] = cls_score
        else:
            target_pos, target_sz, _ = self.update(
                x_crop,
                target_pos,
                target_sz * scale_z,
                window,
                scale_z,
                p,
                debug=debug,
                writer=writer,
            )

        target_pos[0] = max(0, min(state["im_w"], target_pos[0]))
        target_pos[1] = max(0, min(state["im_h"], target_pos[1]))
        target_sz[0] = max(10, min(state["im_w"], target_sz[0]))
        target_sz[1] = max(10, min(state["im_h"], target_sz[1]))

        # print("cropped x shape: ", x_crop.shape)
        # print("target pos shape: ", target_pos.shape)
        # print("target size shape: ", target_sz.shape)
        # print("target size: ", target_sz)

        # TODO: compute appropriate bounding box in x,y,w,h format (?) and return it
        location = cxy_wh_2_rect(target_pos, target_sz)

        # set lighttrack params
        state["target_pos"] = target_pos
        state["target_sz"] = target_sz
        state["p"] = p

        # set pytracking params
        state["target_bbox"] = location
        # print("location: ", location)
        return state

    def grids(self, p):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        # print('ATTENTION',p.instance_size,p.score_size)
        sz = p.score_size  # 16

        # the real shift is -param['shifts']
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(
            np.arange(0, sz) - np.floor(float(sz_x)),
            np.arange(0, sz) - np.floor(float(sz_y)),
        )

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

    def change(self, r):
        return np.maximum(r, 1.0 / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def extract_backbone_features(self, im):
        layers = self.params.backbone_feature_layer

        with torch.no_grad():
            # backbone_features = self.net.backbone_net(im.cuda(), layers)
            backbone_features = self.net.backbone_net(im.to(self.params.device), layers)
            # backbone_features = self.net.backbone_net(im, layers)
        return backbone_features

    def normalize(self, x):
        """input is in (C,H,W) format"""
        x /= 255
        x -= self.mean
        x /= self.std
        return x


class Config(object):
    def __init__(self, stride=8, even=1):
        self.penalty_k = 0.007  # 0.062
        self.window_influence = 0.225  # 0.38
        self.lr = 0.616  # 0.765
        self.windowing = "cosine"
        if even:
            self.exemplar_size = 128
            self.instance_size = 256
        else:
            self.exemplar_size = 127
            self.instance_size = 255
        # total_stride = 8
        # score_size = (instance_size - exemplar_size) // total_stride + 1 + 8  # for ++
        self.total_stride = stride
        self.score_size = int(round(self.instance_size / self.total_stride))
        self.context_amount = 0.5
        self.ratio = 1  # 0.94

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        # self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8 # for ++
        self.score_size = int(round(self.instance_size / self.total_stride))
