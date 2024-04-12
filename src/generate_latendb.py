#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import datasets
from models import LATENT_ENCODER, CLASS_ENCODER
from utils import LOGGER, DATABASEDIR, CPU_COUNT, BSGuard, DEVICE
from utils.dataloader import MonkeyLoader
from utils.transformations import NONE


def save_hist_plot(raw, title=''):
    hist = (raw == raw.max(dim=1, keepdim=True)[0]).to(torch.float32).sum(dim=0)
    hist /= hist.sum()
    hist_frame = pd.DataFrame(hist.numpy())
    hist_frame.plot()
    plt.legend().set_visible(False)
    plt.xlabel('Omniglot class-index')
    plt.ylabel('Occurrences in %')
    plt.title(title)
    plt.savefig(os.path.join(DATABASEDIR, title, 'omni_hist.png'))
    return hist


@BSGuard
def latent_encode(encoder, loader, dset_name):
    encodings = []
    loader.dataset.transform = NONE()
    for d, t in tqdm(loader):
        d = d.to(DEVICE, dtype=torch.float32, non_blocking=True)
        en = encoder(d)
        encodings.append(en.detach().cpu())
        del en
    encodings = torch.cat(encodings, dim=0).squeeze()
    torch.save(encodings, os.path.join(DATABASEDIR, dset_name, 'encodings.pt'))


@BSGuard
def class_encode(encoder, loader, dset_name):
    classhist = []
    for d, t in tqdm(loader):
        d = d.to(DEVICE, dtype=torch.float32, non_blocking=True)
        hist = encoder(d)
        classhist.append(hist.detach().cpu())
        del hist
    classhist = torch.cat(classhist, dim=0).squeeze()
    save_hist_plot(classhist, dset_name)
    torch.save(classhist, os.path.join(DATABASEDIR, dset_name, 'classhist.pt'))


def get_loader(dset_name):
    dataset_args = {}
    if dset_name == 'EMNIST':
        dataset_args = {'split': 'balanced'}
    dset = getattr(datasets, dset_name)
    train = dset(
        root=DATABASEDIR, transform=NONE(), train=True, download=True, **dataset_args
    )
    return MonkeyLoader(
        dataset=train,
        batch_size=100,
        shuffle=False,
        pin_memory=True,
        num_workers=CPU_COUNT
    )


if __name__ == '__main__':
    """
    This is just an example of how you can use train and evaluate
    to interact with the configurable network
    """
    cmdline_parser = argparse.ArgumentParser('AutoML SS19 final project')

    cmdline_parser.add_argument(
        '-v', '--verbosity', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity'
    )
    args, unknowns = cmdline_parser.parse_known_args()

    log_lvl = logging.DEBUG if args.verbosity == 'DEBUG' else logging.INFO
    LOGGER.setLevel(log_lvl)

    if unknowns:
        LOGGER.warning('Found unknown arguments!')
        LOGGER.warning(str(unknowns))
        LOGGER.warning('These will be ignored')

    lt_en = LATENT_ENCODER.to(DEVICE, non_blocking=True)
    cl_en = CLASS_ENCODER.to(DEVICE, non_blocking=True)
    for dset in ['KMNIST', 'EMNIST', 'QMNIST', 'MNIST', 'Omniglot']:
        tl = get_loader(dset)
        LOGGER.info(f'Start generating class encoding of "{dset}"')
        class_encode(cl_en, tl, dset)
        LOGGER.info(f'Finished')
        LOGGER.info(f'Start generating latent encoding of "{dset}"')
        latent_encode(lt_en, tl, dset)
        LOGGER.info(f'Finished')
