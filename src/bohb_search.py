#!/usr/bin/env python
import os
import copy
import random
import time
import traceback
import pprint
import glob
import json
import torch
import numpy as np
import threading
# import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import DEVICE, LOGDIR, LOGGER, CHECKPOINTDIR, Config

# import hpbandster.visualization as hpvis
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from configspaces import K49CS, BASELINECNN_CFG, RESNET18_CFG  # noqa
from trainer import Trainer


def train_with_config(config):
    t = Trainer(config)
    return t.run()


class InvalidConfigurationError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BOHBWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pp = pprint.PrettyPrinter(indent=4)

    def format_config(self, config, budget):
        tmp = config.copy()
        tmp.update({'epochs': int(budget)})

        if np.any(['model_args.' in k for k in config.keys()]):
            model_args = {
                k.split('.')[-1]: v
                for k, v in config.items()
                if 'model_args' in k and tmp.pop(k) and k.split('.')[-1] != ''
            }
            tmp.update({'model_args': dict(model_args)})

        if np.any(['optim_args.' in k for k in config.keys()]):
            optim_args = {
                k.split('.')[-1]: v
                for k, v in config.items()
                if 'optim_args' in k and tmp.pop(k) and k.split('.')[-1] != ''
            }
            tmp.update({'optim_args': dict(optim_args)})

        if np.any(['transform_args.' in k for k in config.keys()]):
            transform_args = {
                k.split('.')[-1]: v
                for k, v in config.items()
                if 'transform_args' in k and tmp.pop(k) and k.split('.')[-1] != ''
            }
            tmp.update({'transform_args': dict(transform_args)})

        return tmp

    def check_constraints(self, config):
        if config['optimizer'] == 'Adam' and config['optim_args']['nesterov']:
            raise InvalidConfigurationError('Vanilla Adam does not support nesterov')
        if config['model'] == 'BaselineCNN' and config['model_args']['zero_init_residual']:
            raise InvalidConfigurationError(
                'BaselineCNN does not support zero_init_residual'
            )
        if config['model'] == 'BaselineCNN' and config['warmstart']:
            raise InvalidConfigurationError('BaselineCNN does not support warmstart')
        if config['warmstart'] and config['transforms'] != 'OCR':
            raise InvalidConfigurationError(
                'Warmstart requires OCR transformations, namely resize'
            )

    def compute(self, config, budget, *args, **kwargs):
        base_config = copy.deepcopy(
            BASELINECNN_CFG if config['model'] == 'BaselineCNN' else RESNET18_CFG
        )
        run_config = Config.merge_dicts(base_config, config, False)
        run_config['epochs'] = int(budget)

        LOGGER.info('START BOHB ITERATION')
        LOGGER.info(f'CONFIG:\n{self.pp.pformat(run_config)}')
        LOGGER.info(f'BUDGET: {budget}')

        try:
            self.check_constraints(run_config)
        except InvalidConfigurationError as e:
            LOGGER.warning(e)
            raise e

        val_score, test_score = -1e10, -1e10
        try:
            val_score, test_score = train_with_config(run_config)
        except Exception as e:
            status = traceback.format_exc()
            LOGGER.warning(status)
            raise e

        LOGGER.info(f'FINAL SCORE: {-val_score}')
        return {'loss': -val_score, 'info': {'test_error': -test_score}}


#######################################################################################################################
def get_best_config():
    best_conf = None
    best_checkpoint_fp = None
    filefilter = os.path.join(CHECKPOINTDIR, 'K49', '*.json')
    for fp in glob.glob(filefilter):
        with open(fp, 'r') as f:
            conf = json.load(f)
        if best_conf is None or conf['val_score'] > best_conf['val_score']:
            best_conf = conf
            best_checkpoint_fp = fp.replace('.json', '.pth')
    return best_conf, best_checkpoint_fp


def runBOHB(iterations, timeout):
    run_id = "0"

    # assign random port in the 30000-40000 range to avoid using a blocked
    # port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(nameserver="127.0.0.1", run_id=run_id, nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=LOGDIR + '_bohb', overwrite=True)

    bohb = BOHB(
        configspace=K49CS,
        run_id=run_id,
        min_budget=10,
        max_budget=20,
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger,
    )

    bohb_thread = threading.Thread(target=bohb.run, kwargs={'n_iterations': iterations})
    bohb_thread.start()
    bohb_start = time.time()
    elapsed = time.time() - bohb_start
    while bohb_thread.is_alive() and elapsed < timeout:
        time.sleep(1)
        elapsed = time.time() - bohb_start
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    runBOHB(9001, 24 * 60 * 60)
    config, path = get_best_config()
    config['epochs'] = 0
    config['warmstart'] = False
    config['autotune_lr'] = False
    config['optim_args']['lr'] = 0

    trainer = Trainer(config)
    trainer.model.main_net.load_state_dict(torch.load(path))
    trainer.model.eval()

    t = tqdm(trainer.test_loader)
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in t:
            images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = trainer.model(images)
            predictions.append(outputs.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
            del outputs
    predictions = np.vstack(predictions)
    targets = np.hstack(targets)
    torch.save(
        predictions,
        os.path.join(CHECKPOINTDIR, 'K49', path.replace('.pth', '_predictions.pt'))
    )
    torch.save(
        targets, os.path.join(CHECKPOINTDIR, 'K49', path.replace('.pth', '_targets.pt'))
    )
