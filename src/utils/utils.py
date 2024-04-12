import atexit
import signal
import logging
import json
import os
import re
import sys
import time
import webbrowser
import multiprocessing
from collections import OrderedDict
from functools import partial, wraps

import hjson
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# #############################################################################
#                                PATHS
# #############################################################################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHECKPOINTDIR = os.path.abspath(os.path.join(BASEDIR, os.pardir, 'checkpoints'))
DATABASEDIR = os.path.abspath(os.path.join(BASEDIR, os.pardir, 'data'))
TORCH_HOME = os.path.abspath(os.path.join(BASEDIR, os.pardir, 'torchhome'))
os.environ['TORCH_HOME'] = TORCH_HOME

# #############################################################################
#                                CONST
# #############################################################################
CPU_COUNT = 0  # torch.multiprocessing.cpu_count()
KB = 1024
MB = 1024**2
GB = 1025**3
KEEP_AVAILABLE = 3 * GB


# #############################################################################
#                                GENERAL CONFIG
# #############################################################################
class Config:
    @classmethod
    def merge_dicts(_, d, side_conf, inplace=True):
        d = d if inplace else d.copy()
        for p, v in side_conf.items():
            tmp_dict = d
            p_split = p.split('.')
            for stub in p_split[:-1]:
                if stub not in tmp_dict:
                    tmp_dict[stub] = OrderedDict()
                tmp_dict = tmp_dict[stub]
            leaf = p_split[-1]
            if leaf in tmp_dict and not isinstance(tmp_dict[leaf], type(v)):
                LOGGER.warning(
                    'Overriding config at "{}" value with another type! {} => {}'.format(
                        p, type(tmp_dict[leaf]), type(v)
                    )
                )
            tmp_dict[leaf] = v
        return d

    def __init__(self, config_path):
        with open(config_path) as config_file:
            self.__dict__ = hjson.load(config_file)

    def write(self, save_path):
        with open(save_path, 'w') as save_file:
            save_file.write(hjson.dumps(self.__dict__))

    def side_load_config(self, sideload_conf_path):
        '''
        This overrides all configs defined in the sideload_config.
        To target specific leafs in a hierarchy use '.' to separate
        the parent in the leafs' path:
        ie. { selection.video.optim_args.lr: 0.1} would overwrite the
        video optimizer's initial learning rate
        In addition paths and leafs not found in config are added

        ATTENTION: The sideload_config.json can define dictionaries as values as well
        so it is possible to overwrite a whole subhierarchy.
        '''

        if not os.path.isfile(sideload_conf_path):
            return
        with open(sideload_conf_path, 'r') as file:
            side_conf = json.load(file)

        self.merge_dicts(self.__dict__, side_conf)


CONFIG = Config(os.path.join(BASEDIR, 'config.hjson'))
CONFIG.side_load_config(os.path.join(BASEDIR, 'sideload_config.hjson'))

# #############################################################################
#                                SEEDS
# #############################################################################
np.random.seed(CONFIG.np_random_seed)
torch.manual_seed(CONFIG.torch_manual_seed)
try:
    torch.cuda.manual_seed_all(CONFIG.torch_cuda_manual_seed_all)
    torch.backends.cudnn.benchmark = CONFIG.cudnn_benchmark
    torch.backends.cudnn.deterministic = CONFIG.cudnn_deterministic
except Exception:
    pass

# #############################################################################
#                                LOGGING
# #############################################################################
LOGDIR = os.path.abspath(
    os.path.join(
        BASEDIR, os.pardir, 'logs', time.strftime("%Y%m%d_%H%M%S", time.localtime())
    )
)
if (
    not os.path.isdir(LOGDIR) and CONFIG.log_to_file and
    multiprocessing.current_process().name == 'MainProcess'
):
    os.mkdir(LOGDIR)


class ColoredFormatter(logging.Formatter):
    GREY = '\033[;90m'
    YELLOW = '\033[;93m'
    RED = '\033[;91m'
    BOLD = '\033[;1m'
    ENDC = '\033[;0m'

    def format(self, record):
        s = super(ColoredFormatter, self).format(record)
        if record.levelno == logging.DEBUG:
            s = self.GREY + s + self.ENDC
        elif record.levelno == logging.WARNING:
            s = self.YELLOW + s + self.ENDC
        elif record.levelno == logging.ERROR:
            s = self.RED + s + self.ENDC
        elif record.levelno == logging.CRITICAL:
            s = self.BOLD + self.RED + s + self.ENDC + self.ENDC
        return s


def get_logger():
    '''
    Set logging format to something like:
            2019-04-25 12:52:51,924 INFO model.py: <message>
    '''
    class LessThanFilter(logging.Filter):
        def __init__(self, exclusive_maximum, name=''):
            super(LessThanFilter, self).__init__(name)
            self.max_level = exclusive_maximum

        def filter(self, record):
            # non-zero return means we log this message
            return 1 if record.levelno < self.max_level else 0

    formatting = '%(asctime)s %(levelname)s %(filename)s: %(message)s'

    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, CONFIG.log_level)
    logger.setLevel(logging_level)

    if CONFIG.log_to_file and multiprocessing.current_process().name == 'MainProcess':
        fileout_handler = logging.FileHandler(os.path.join(LOGDIR, 'run.log'), mode='w')
        fileout_handler.setLevel(logging.DEBUG)
        fileout_handler.setFormatter(logging.Formatter(fmt=formatting))
        logger.addHandler(fileout_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.addFilter(LessThanFilter(logging.WARNING))
    stdout_handler.setFormatter(ColoredFormatter(fmt=formatting))
    logger.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(ColoredFormatter(fmt=formatting))
    logger.addHandler(stderr_handler)

    logger.propagate = False
    return logger


LOGGER = get_logger()

# #############################################################################
#                                TENSORBOARD
# #############################################################################
try:
    if not CONFIG.tensorboard_logging:
        raise Exception
    from tensorboardX import SummaryWriter
    SW = SummaryWriter(LOGDIR)

    tboard_cmd = 'tensorboard --logdir ' + LOGDIR
    with open(os.path.join(LOGDIR, 'start_tboard.sh'), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('xdg-open http://localhost:6006/\n')
        f.write(tboard_cmd + '\n')
    os.chmod(os.path.join(LOGDIR, 'start_tboard.sh'), 0o744)

    def kill_if_alive(p, msg):
        LOGGER.info(msg)
        p.terminate()

    if CONFIG.tensorboard_launch:
        tbp = psutil.Popen([*tboard_cmd.split(' ')], shell=False)
        webbrowser.open_new_tab('http://localhost:6006/')
        atexit.register(
            kill_if_alive, tbp,
            'Terminating tensorboard with PID \033[92m{}\033[0m'.format(tbp.pid)
        )
        LOGGER.info('Started tensorboard with PID \033[92m{}\033[0m'.format(tbp.pid))

    # Make atexit work for terminate cases
    def sigHandler(sig_no, sig_frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigHandler)

except Exception:

    class BlackHole():
        '''
        Has everything, swallows all and returns nothing...
        and very handy to replace non-existing loggers but keeping
        the lines of code for it the same
        '''
        def __getattr__(self, attr):
            def f(*args, **kwargs):
                return

            return f

        def __setattr__(self, attr, val):
            return

    SW = BlackHole()

# #############################################################################
#                                MEM PROFILING
# #############################################################################
try:
    if not CONFIG.profile_mem:
        raise Exception
    from memory_profiler import profile as memprofile
    fp = open(os.path.join(LOGDIR, 'memory_profiler.log'), 'w+')
    memprofile = partial(memprofile, stream=fp)
except Exception:
    # Fake version of the memory_profiler's profile decorator
    def memprofile(func=None, stream=None, precision=1, backend='psutil'):
        '''
        Stripped version of memory_profile's profile decorator
        '''
        if func is not None:

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper
        else:

            def inner_wrapper(f):
                return memprofile(f, stream=stream, precision=precision, backend=backend)

            return inner_wrapper


# #############################################################################
#                                PRINTOUTS
# #############################################################################
def print_vram_usage():
    LOGGER.info(
        'MAX VRAM ALLOCATED:\t{} MB'.format(torch.cuda.max_memory_allocated() / MB)
    )
    LOGGER.info('MAX VRAM CACHED:\t\t{} MB'.format(torch.cuda.max_memory_cached() / MB))


# #############################################################################
#                                CUDA OOM GUARD
# #############################################################################
def parse_cumem_error(err_str):
    mem_search = re.search(
        r'Tried to allocate ([0-9].*? [G|M])iB.*\; ([0-9]*.*? [G|M])iB free', err_str
    )
    tried, free = mem_search.groups()
    tried = float(tried[:-2]) * 1024 if tried[-1] == 'G' else float(tried[:-2])
    free = float(free[:-2]) * 1024 if free[-1] == 'G' else float(free[:-2])
    return tried, free


def BSGuard(f):
    '''
    Decorator to safe-guard the execution of f against cuda's out of memory error
    and to recover from it if possible by reducing the batch-size of the given
    dataloader
    '''
    @wraps(f)
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except RuntimeError as e:
                if ('CUDA out of memory.' not in e.args[0]):
                    raise e
                LOGGER.warn('CAUGHT VMEM ERROR! SCALING DOWN BATCH-SIZE!')
                try:
                    loader = [
                        v for v in [*args, *kwargs.values()] if isinstance(v, DataLoader)
                    ][0]
                except IndexError as e:
                    LOGGER.error('Could not find dataloader in local context')
                    raise e
                if (loader.batch_size == 1):
                    LOGGER.error('Cannot reduce batch_size lower than 1')
                    raise e
                tried_mem, free_mem = parse_cumem_error(e.args[0])
                # Not working as intended. I blame fragmantation!
                # mem_downscale = min(1, free_mem / (tried_mem + 1e-8)) if free_mem < tried_mem else 0.9
                mem_downscale = 0.75
                loader.batch_size = max(1, int(loader.batch_size * mem_downscale))
                LOGGER.warn(
                    'TRIED TO ALLOCATE {0:.2f} MiB WITH {1:.2f} MiB FREE'.format(
                        tried_mem, free_mem
                    )
                )
                LOGGER.warn('BATCH-SIZE NOW IS {}'.format(loader.batch_size))

    return decorated


# #############################################################################
#                                TORCH MODULE HELPERS
# #############################################################################
class AugmentNet(nn.Module):
    '''
    The augment net's purpose is to perform augmentations on a whole batch at once
    on the GPU instead of per element on the cpu. If this has benefit still needs
    to be determined
    '''
    def __init__(self, transfs):
        super().__init__()
        self.augmentation = {
            'train': (nn.Sequential(*transfs['train'])),
            'eval': (nn.Sequential(*transfs['test']))
        }

    def forward(self, x):
        mode = 'train' if self.training else 'eval'
        x = self.augmentation[mode](x)
        return x


class MonkeyNet(nn.Sequential):
    '''
    The idea of the monkeynet is to expose all attributes of the networks
    it's given and is therefore a special kind of Sequential network
    '''
    def __init__(self, *nets):
        super().__init__(*nets)
        super().__setattr__('__finished_init__', True)

    def __getattr__(self, attr):
        try:
            ret = super(nn.Sequential, self).__getattribute__(attr)
        except AttributeError:
            super(nn.Sequential, self).__getattribute__('__finished_init__')
        try:
            ret = super(nn.Sequential, self).__getattr__(attr)
        except AttributeError:
            pass
        for m in self.children():
            try:
                ret = getattr(m, attr)
            except AttributeError:
                continue
        if 'ret' not in locals():
            raise AttributeError(
                'The monkey is sorry because it could not find '
                '{0}'
                ''.format(attr)
            )
        else:
            return ret

    def __setattr__(self, attr, val):
        try:
            super(nn.Sequential, self).__getattribute__(attr)
            super(nn.Sequential, self).__setattr__(attr, val)
            return
        except AttributeError:
            try:
                super(nn.Sequential, self).__getattribute__('__finished_init__')
            except AttributeError:
                super(nn.Sequential, self).__setattr__(attr, val)
                return
        for m in self.children():
            try:
                getattr(m, attr)
                setattr(m, attr, val)
                return
            except AttributeError:
                continue
        raise AttributeError(
            'The monkey is sorry because it could not set '
            '{0}'
            ''.format(attr)
        )
