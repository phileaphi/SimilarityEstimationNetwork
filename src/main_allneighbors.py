#!/usr/bin/env python
import argparse
import logging
import json
import torch

from trainer import Trainer
from utils import LOGGER, CHECKPOINTDIR  # noqa


def train_with_config(config):
    t = Trainer(config)
    return t.run()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network
    """
    cmdline_parser = argparse.ArgumentParser('AutoML SS19 final project')
    cmdline_parser.add_argument(
        '-c',
        '--config',
        help='Path to the config to use. Configs are expected in the json format.',
        required=True
    )
    cmdline_parser.add_argument(
        '-v', '--verbosity', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity'
    )
    args, unknowns = cmdline_parser.parse_known_args()

    log_lvl = logging.DEBUG if args.verbosity == 'DEBUG' else logging.INFO
    LOGGER.setLevel(log_lvl)

    cfg = None
    if unknowns:
        LOGGER.warning('Found unknown arguments!')
        LOGGER.warning(str(unknowns))
        LOGGER.warning('These will be ignored')
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg = json.load(f)

    for d in ['KMNIST', 'EMNIST', 'QMNIST', 'MNIST', 'Omniglot']:
        cfg['neighbor_set'] = d
        val_score, test_score = train_with_config(cfg)
    LOGGER.info(f'Validation Score: {val_score:.4f}\tTest Score: {test_score:.4f}')
