import os
import json
from functools import wraps
from collections import OrderedDict

import torch

from .mae import MAE
from .resnet import Resnet
from utils import CHECKPOINTDIR, LOGGER, MonkeyNet, AugmentNet
from utils.transformations import Grayscale, Interpolate, Clamp, Normalize, BinarizeImage

__all__ = ['LATENT_ENCODER', 'CLASS_ENCODER']

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def mae_forward(input):
    en, _, _ = mae.encode(input, 1)
    return en


def LOAD_ON_DEMAND(func=None, model=None):
    if func is not None:

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not model.checkpoint_loaded:
                if isinstance(model, MAE):
                    LOGGER.info('Loading encoder checkpoint for LATENT ENCODER')
                    mae_state_dict = torch.load(
                        os.path.join(CHECKPOINTDIR, 'Omniglot', 'MAE', 'model.pt')
                    )
                    mae.load_state_dict(mae_state_dict)
                else:
                    LOGGER.info('Loading encoder checkpoint for CLASS ENCODER')
                    model._load_best_known_for('Omniglot')

                setattr(model, 'checkpoint_loaded', True)
                model.eval()
            return func(*args, **kwargs)

        return wrapper
    else:

        def inner_warpper(f):
            return LOAD_ON_DEMAND(f, model=model)

        return inner_warpper


mae_params = json.load(
    open(os.path.join(CHECKPOINTDIR, 'Omniglot', 'MAE', 'config.json'), 'r')
)
mae = MAE.from_params(mae_params)
mae.forward = mae_forward
setattr(mae, 'checkpoint_loaded', False)
mae.forward = LOAD_ON_DEMAND(mae.forward, mae)
aug_net = AugmentNet(
    {
        'train':
            [
                Grayscale(),
                Interpolate((28, 28)),
                Clamp(0, 255),
                Normalize(),
                BinarizeImage(),
            ],
        'test':
            [
                Grayscale(),
                Interpolate((28, 28)),
                Clamp(0, 255),
                Normalize(),
                BinarizeImage(),
            ]
    }
)
LATENT_ENCODER = MonkeyNet(OrderedDict([
    ('aug_net', aug_net),
    ('main_net', mae),
]))

resnet = Resnet(
    block='BasicBlock',
    layers=[2, 2, 2, 2],
    input_shape=(1, 64, 64),
    num_classes=1623,
)
setattr(mae, 'checkpoint_loaded', False)
resnet.forward = LOAD_ON_DEMAND(resnet.forward, resnet)
aug_net = AugmentNet(
    {
        'train': [
            Grayscale(),
            Interpolate((64, 64)),
            Clamp(0, 255),
            Normalize(),
        ],
        'test': [
            Grayscale(),
            Interpolate((64, 64)),
            Clamp(0, 255),
            Normalize(),
        ]
    }
)
CLASS_ENCODER = MonkeyNet(OrderedDict([
    ('aug_net', aug_net),
    ('main_net', resnet),
]))

LATENT_ENCODER.eval()
CLASS_ENCODER.eval()
