import json
import copy
import os
import time
import sys
import glob
import itertools
import inspect
import matplotlib.pyplot as plt

from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from torchsummary import summary
from tqdm import tqdm

import datasets
import models
from models import CLASS_ENCODER, LATENT_ENCODER  # noqa
from utils import BSGuard, DEVICE, LOGGER, CHECKPOINTDIR, DATABASEDIR, CPU_COUNT, AugmentNet, MonkeyNet, SW
from utils import optimizers, loss, transformations
from utils.transformations import NONE
from utils.dataloader import MonkeyLoader
from utils.metrics import AvgrageMeter, accuracy


class Trainer():
    def __init__(self, config):
        resultdir = os.path.join(CHECKPOINTDIR, config['dataset'])
        if os.path.isfile(resultdir):
            os.remove(CHECKPOINTDIR, config['dataset'])
        if not os.path.isdir(resultdir):
            os.mkdir(resultdir)
        self.resultbasepath = os.path.join(
            resultdir,
            config['model'] + '_' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
        )
        self.encoder = CLASS_ENCODER
        self.input_shape = None
        self.loaders = None
        self.test_loader = None

        self.config = config
        self.neighbor_set = config['neighbor_set'] if 'neighbor_set' in config else None
        self.kfold = config['kfold'] if 'kfold' in config else 1
        self.model = getattr(models, config['model'])
        self.optimizer = getattr(optimizers, config['optimizer'])
        self.loss_fn = getattr(loss, config['loss_fn'])
        self.transforms = getattr(transformations, config['transforms'])
        self.num_epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.dataset = getattr(datasets, config['dataset'])
        self.dataset_args = {}
        if self.config['dataset'] == 'EMNIST':
            self.dataset_args = {'split': 'balanced'}

        autotune_lr = config['autotune_lr']
        if autotune_lr:
            config['optim_args']['lr'] = 0

        self._setup()

        self._setup_loaders(self.train_dataset, self.test_dataset, self.kfold)
        LOGGER.info(f'Using {CPU_COUNT} workers for dataloading')

        if autotune_lr:
            self._autotune_lr()
            config['optim_args']['lr'] = self.optimizer.param_groups[0]['lr']

        self.save_result = config.pop('save_result', True)
        self.tb_ns = (
            config['neighbor_set']
            if 'neighbor_set' in config and config['neighbor_set'] is not None else ''
        )

    def _get_known_args(self, func, kwargs):
        k_kwargs = inspect.getargspec(func).args
        res = {k: v for k, v in kwargs.items() if k in k_kwargs}
        return res

    def _setup(self):
        self.train_dataset = self.dataset(
            root=DATABASEDIR,
            train=True,
            transform=self.transforms(train=True, **self.config['transform_args']),
            download=True,
            **self.dataset_args
        )
        self.test_dataset = self.dataset(
            root=DATABASEDIR,
            train=False,
            transform=self.transforms(train=False, **self.config['transform_args']),
            download=True,
            **self.dataset_args
        )

        # Measure dataset distance and give the dataset
        # closest to the model load the respective pretrained
        # weights
        neighbor_set = self.neighbor_set
        if (
            neighbor_set is None and 'warmstart' in self.config and
            self.config['warmstart']
        ):
            neighbor_set = self._get_neighborset(self.train_dataset)

        d_shape = next(iter(self.train_dataset))[0].shape
        self.model = self.model(
            input_shape=d_shape,
            num_classes=len(self.train_dataset.classes),
            neighbor_set=neighbor_set,
            **self._get_known_args(self.model, self.config['model_args']),
        )
        resize_target = (
            self.config['transform_args']['resize_target']
            if 'resize_target' in self.config['transform_args'] else d_shape[1]
        )
        aug_net = AugmentNet(
            {
                'train':
                    [
                        transformations.Grayscale(),
                        transformations.Interpolate(resize_target),
                        transformations.Clamp(0, 255),
                        transformations.Normalize(),
                    ],
                'test':
                    [
                        transformations.Grayscale(),
                        transformations.Interpolate(resize_target),
                        transformations.Clamp(0, 255),
                        transformations.Normalize(),
                    ],
            }
        )
        self.model = MonkeyNet(
            OrderedDict([
                ('aug_net', aug_net),
                ('main_net', self.model),
            ])
        )
        # Something strange going on if
        # pretrained moved to gpu in train mode
        self.model.eval()
        self.model = self.model.to(DEVICE, non_blocking=True)

        self.optimizer = self.optimizer(
            self.model.parameters(),
            **self._get_known_args(self.optimizer, self.config['optim_args']),
        )

        self.loss_fn = self.loss_fn().to(DEVICE, non_blocking=True)
        try:
            LOGGER.info('Generated Network:')
            summary(self.model, (d_shape[0], d_shape[1], d_shape[2]), device=DEVICE.type)
        except Exception:
            LOGGER.warning('Summary ended unexpectedly.')
        LOGGER.info(f'Input Shape: {d_shape}')

    def _get_neighborset(self, train_dataset):
        @BSGuard
        def class_encode(enc, loader, out):
            tmp = []
            for d, _ in tqdm(loader):
                d = d.to(DEVICE, dtype=torch.float32, non_blocking=True)
                en = enc(d)
                tmp.append(en.detach().cpu())
                del en
            out.extend(tmp)

        def get_hist(raw, save_plot=False, title=''):
            hist = (raw == raw.max(dim=1, keepdim=True)[0]).to(torch.float32).sum(dim=0)
            hist /= hist.sum()
            if save_plot:
                hist_frame = pd.DataFrame(hist.numpy())
                hist_frame.plot()
                plt.legend().set_visible(False)
                plt.xlabel('Omniglot class-index')
                plt.ylabel('Occurrences in %')
                plt.title(title)
                plt.savefig(self.resultbasepath + '_hist_' + title.lower() + '.png')
            return hist

        encoder = self.encoder.to(DEVICE, non_blocking=True)
        encodings = []
        old_transform = train_dataset.transform
        train_dataset.transform = NONE()
        loader = MonkeyLoader(
            dataset=train_dataset,
            batch_size=200,
            shuffle=True,
            pin_memory=True,
            num_workers=CPU_COUNT
        )
        encodings_path = os.path.join(DATABASEDIR, self.config['dataset'], 'classhist.pt')
        if not os.path.isfile(encodings_path):
            LOGGER.info('No encoding found. Start encoding train data')
            class_encode(encoder, loader, encodings)
            encodings = torch.cat(encodings, dim=0)
            torch.save(encodings, encodings_path)
        else:
            LOGGER.info('Encoding was already generated. Loading encoding')
            encodings = torch.load(encodings_path)

        train_dataset.transform = old_transform
        target_hist = get_hist(encodings, True, 'Target')

        LOGGER.info('Searching for closest support set')
        min_distance = float('inf')
        neighbor_set = 'Omniglot'
        for classhist_fp in glob.glob(os.path.join(DATABASEDIR, '*', 'classhist.pt')):
            dataset_name = os.path.basename(os.path.dirname(classhist_fp))
            if dataset_name == self.config['dataset']:
                continue
            raw = torch.load(classhist_fp)
            hist = get_hist(raw, False, dataset_name)
            distance = (target_hist - hist).abs().sum()
            LOGGER.debug(f'{dataset_name} -> Target: {distance}')
            if distance < min_distance:
                min_distance = distance
                neighbor_set = dataset_name
        LOGGER.info('Closest support set found:')
        LOGGER.info(f'{neighbor_set} -> Target: {min_distance}')
        del encoder
        return neighbor_set

    def _setup_loaders(self, train_dataset, test_dataset, k=1):
        if k == 1:
            lengths = (np.array([0.15, 0.85]) *
                       len(train_dataset)).round().astype(int).tolist()
            splits = random_split(train_dataset, lengths)
        else:
            lengths = np.floor(k * [len(train_dataset) / k]).astype(int).tolist()
            for i in range(len(train_dataset) - np.sum(lengths)):
                lengths[i] += 1
            splits = random_split(train_dataset, lengths)
        self.loaders = [
            MonkeyLoader(
                dataset=s,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=CPU_COUNT,
            ) for s in splits
        ]

        self.test_loader = MonkeyLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=CPU_COUNT
        )

    def _autotune_lr(self):
        """
        Refined gridsearch for the learning rate on a subset to find a near
        optimal initial learning rate, inspired by 3.3 in:

        No More Pesky Learning Rate Guessing Games
        Leslie N. Smith et al.
        https://arxiv.org/pdf/1506.01186v2.pdf

        This algorithm evaluates all magnitudes between 1e-9 and 1e-0 and picks
        the best one for the next search. The next gridsearch searches the space
        between the left and right neighbor of the currently best lr and uses 20
        evaluations. This is repeated one more time before the currently best lr
        is returned.

        As this execution is protected by the utils/utils.py:BSGuard the biggest
        feasible batch_size is also generated
        """
        def _eval_lrs(losses, vals, initial_model_sd, initial_optim_sd, loader):
            try:
                for lr in vals:
                    self.model.load_state_dict(initial_model_sd)
                    self.optimizer.load_state_dict(initial_optim_sd)
                    self._set_lr(lr)
                    LOGGER.debug(f'Testing LR "{lr:.4e}')
                    _, train_loss = self.train_fn(loader, silent=True)
                    losses = losses.append(
                        {
                            'LR': lr,
                            'Loss': train_loss
                        }, ignore_index=True
                    )
            except ValueError:
                pass
            return losses

        LOGGER.info('Running Autotune LR')
        initial_model_sd = copy.deepcopy(self.model.state_dict())
        initial_optim_sd = copy.deepcopy(self.optimizer.state_dict())

        subset_size = int(len(self.loaders[0].dataset) * 0.05)
        num_batches = int(np.ceil(subset_size / self.batch_size))
        entries = [
            k for k in itertools.islice(itertools.chain(*self.loaders), 0, num_batches)
        ]

        ds, ts = (
            torch.cat([k[0] for k in entries], dim=0),
            torch.cat([k[1] for k in entries], dim=0),
        )
        toyset = torch.utils.data.TensorDataset(ds[:subset_size], ts[:subset_size])
        loader = MonkeyLoader(
            dataset=toyset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
        losses = pd.DataFrame(columns=['LR', 'Loss'])
        lower, upper = [], []

        vals = [0]
        losses = _eval_lrs(losses, vals, initial_model_sd, initial_optim_sd, loader)

        vals = np.logspace(-9., 0., num=10)
        losses = _eval_lrs(losses, vals, initial_model_sd, initial_optim_sd, loader)
        best = losses.loc[losses['Loss'].idxmin()]

        best_idx = losses[losses['LR'] == best['LR']].index[0]
        if best_idx > 0:
            lower = np.linspace(
                losses.loc[best_idx - 1]['LR'],
                best['LR'],
                11,
            )[1:-1]
        if best_idx < len(losses) - 1:
            upper = np.linspace(
                best['LR'],
                losses.loc[best_idx + 1]['LR'],
                11,
            )[1:-1]

        vals = np.concatenate([lower, upper])
        losses = _eval_lrs(losses, vals, initial_model_sd, initial_optim_sd, loader)
        best = losses.loc[losses['Loss'].idxmin()]

        best_idx = losses[losses['LR'] == best['LR']].index[0]
        if best_idx > 0:
            lower = np.linspace(
                losses.loc[best_idx - 1]['LR'],
                best['LR'],
                11,
            )[1:-1]
        if best_idx < len(losses) - 1:
            upper = np.linspace(
                best['LR'],
                losses.loc[best_idx + 1]['LR'],
                11,
            )[1:-1]

        vals = np.concatenate([lower, upper])
        losses = _eval_lrs(losses, vals, initial_model_sd, initial_optim_sd, loader)
        best = losses.loc[losses['Loss'].idxmin()]

        self.model.load_state_dict(initial_model_sd)
        self.optimizer.load_state_dict(initial_optim_sd)
        self._set_lr(best['LR'])
        for l in self.loaders:
            l.batch_size = loader.batch_size
        self.test_loader.batch_size = loader.batch_size

        # Got a bug when running in bohb
        oldout = sys.stdout
        sys.stdout = None
        losses = losses.sort_values('LR')
        losses.plot(
            x='LR',
            y='Loss',
            loglog=True,
        )
        plt.legend().set_visible(False)
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.title('Autotune LR')
        plt.savefig(self.resultbasepath + '_autotune_lr.png')
        sys.stdout = oldout
        LOGGER.debug(losses.sort_values('Loss'))
        LOGGER.info(
            f'Autotune LR finished. New LR is "{best["LR"]:.3e}" with train loss "{best["Loss"]}'
        )

    def _set_lr(self, val):
        for p in self.optimizer.param_groups:
            p['lr'] = val

    def decay_lr(self, decay=1, step_size=0):
        for p in self.optimizer.param_groups:
            p['lr'] *= decay
            p['lr'] -= step_size

    @BSGuard
    def train_fn(self, loader, silent=False):
        """
        Training method
        :return: (accuracy, loss) on the data
        """
        score = AvgrageMeter()
        objs = AvgrageMeter()
        self.model.train()

        t = tqdm(loader) if not silent else loader
        for images, labels in t:
            images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            acc, _ = accuracy(logits, labels, topk=(1, 5))
            n = images.size(0)

            score.update(acc.item(), n)
            objs.update(loss.item(), n)

            if isinstance(t, tqdm):
                t.set_description('(=> Training) Loss: {:.4f}'.format(objs.avg))

        return score.avg, objs.avg

    @BSGuard
    def eval_fn(self, loader, silent=False):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :return: accuracy on the data
        """
        score = AvgrageMeter()
        self.model.eval()

        t = tqdm(loader) if not silent else loader
        with torch.no_grad():
            for images, labels in t:
                images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = self.model(images)
                acc, _ = accuracy(outputs, labels, topk=(1, 5))
                score.update(acc.item(), images.size(0))

                if isinstance(t, tqdm):
                    t.set_description('(=> Evaluation) Score: {:.4f}'.format(score.avg))

        return score.avg

    def run(self):
        # Train the model
        initial_model_sd = copy.deepcopy(self.model.state_dict())
        initial_optim_sd = copy.deepcopy(self.optimizer.state_dict())
        just_eval = self.kfold > 1

        checkpoint_score = -float('inf')
        checkpoint = None

        kval_acc = pd.DataFrame()
        val_score = -float('inf')

        for k in range(self.kfold):
            self.model.load_state_dict(initial_model_sd)
            self.optimizer.load_state_dict(initial_optim_sd)

            val_loader = self.loaders[k]

            train_acc = pd.DataFrame()
            val_acc = pd.DataFrame()
            patience = 0
            epoch = 0
            train_score = -float('inf')
            val_score = -float('inf')

            for epoch in range(self.num_epochs):
                train_loader = self.loaders[0:k] + self.loaders[k + 1:]
                train_loader = itertools.chain(*train_loader)
                LOGGER.info('#' * 50)
                LOGGER.info('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))

                train_score, train_loss = self.train_fn(train_loader)
                LOGGER.info('Train accuracy %f', train_score)

                val_score = self.eval_fn(val_loader)
                LOGGER.info('Validation accuracy %f', val_score)

                SW.add_scalar(self.tb_ns + 'train_loss', train_loss, epoch)
                SW.add_scalar(self.tb_ns + 'train_score', train_score, epoch)
                SW.add_scalar(self.tb_ns + 'val_score', val_score, epoch)
                if val_score > checkpoint_score and self.kfold == 1:
                    checkpoint_score = val_score
                    checkpoint = copy.deepcopy(self.model.state_dict())

                train_acc = train_acc.append([train_score], ignore_index=True)
                val_acc = val_acc.append([val_score], ignore_index=True)
                train_ewm = train_acc.ewm(0.6).mean()
                val_ewm = val_acc.ewm(0.6).mean()

                if len(train_ewm) > 5:
                    if (train_ewm.iloc[-6:-1].mean() > train_ewm.iloc[-1]).all():
                        self.decay_lr(0.2)
                        newlr = self.optimizer.param_groups[0]['lr']
                        LOGGER.info('#' * 50)
                        LOGGER.info(f'Reducing LR. New LR={newlr:.4e}')
                        LOGGER.info('#' * 50)
                        patience += 1
                    if patience > 5 and (
                        val_ewm.iloc[-5:-1].mean() + val_ewm.iloc[-5:-1].std() <
                        val_ewm.iloc[-1]
                    ).all():
                        break

            if self.num_epochs == 0:
                val_score = self.eval_fn(val_loader)
                just_eval = True
            kval_acc = kval_acc.append([val_score], ignore_index=True)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)
            val_score = checkpoint_score

        test_score = self.eval_fn(self.test_loader)
        LOGGER.info('Test accuracy %f', test_score)

        if not just_eval and self.save_result:
            # Save the model checkpoint can be restored via "model = torch.load(CHECKPOINTDIR)"
            torch.save(self.model.main_net.state_dict(), self.resultbasepath + '.pth')
        cfg = self.config.copy()
        with open(self.resultbasepath + '.json', 'w') as f:
            json.dump(cfg, f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')
        cfg.update(
            {
                'warmstart_path': self.model.warmstart_path,
                'eff_batch_size': self.test_loader.batch_size,
                'last_epoch': epoch + 1,
                'val_score': val_score,
                'test_score': test_score,
                'kval_scores': kval_acc.values.squeeze().tolist(),
            }
        )
        with open(self.resultbasepath + '.json', 'w') as f:
            json.dump(cfg, f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')

        return kval_acc.values.mean().tolist(), test_score
