import random

from et_track import ET_Tracker
from tensorlist import TensorList


class TrackerParams:
    """Class for tracker parameters."""

    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)

    def get(self, name: str, *default):
        """Get a parameter value with the given name. If it does not exists, it return the default value given as a
        second argument or returns an error if no default value is given."""
        if len(default) > 1:
            raise ValueError("Can only give one default value.")

        if not default:
            return getattr(self, name)

        return getattr(self, name, default[0])

    def has(self, name: str):
        """Check if there exist a parameter with the given name."""
        return hasattr(self, name)


class FeatureParams:
    """Class for feature specific parameters"""

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError

        for name, val in kwargs.items():
            if isinstance(val, list):
                setattr(self, name, TensorList(val))
            else:
                setattr(self, name, val)


def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = False

    params.checkpoint_epoch = 35

    params.net = ET_Tracker(
        search_size=256,
        template_size=128,
        stride=16,
        e_exemplars=4,
        sm_normalization=True,
        temperature=2,
        dropout=False,
    )

    params.big_sz = 288
    params.small_sz = 256
    params.stride = 16
    params.even = 0
    params.model_name = "et_tracker"

    params.image_sample_size = 256
    params.image_template_size = 128
    params.search_area_scale = 5

    params.window_influence = 0
    params.lr = 0.616
    params.penalty_k = 0.007
    params.context_amount = 0.5

    params.features_initialized = False

    return params
