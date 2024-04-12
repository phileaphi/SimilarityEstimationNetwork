import os
import warnings

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.datasets import MNIST, EMNIST as TV_EMNIST, Omniglot as TV_Omniglot, VisionDataset
from torchvision.datasets.utils import download_url, makedir_exist_ok, verify_str_arg
from torchvision.datasets import *  # noqa # expose all in torch.optim through here


class K49(MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
        'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz',
    ]
    classes = [
        'あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た',
        'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み',
        'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'ゐ', 'ゑ', 'を', 'ん',
        'ゝ'
    ]

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, root=self.raw_folder, filename=filename)

        # process and save as torch files
        print('Processing...')

        training_set = (
            torch.tensor(
                np.load(os.path.join(self.raw_folder, 'k49-train-imgs.npz'))['arr_0']
            ),
            torch.tensor(
                np.load(os.path.join(self.raw_folder, 'k49-train-labels.npz'))['arr_0']
            )
        )
        test_set = (
            torch.tensor(
                np.load(os.path.join(self.raw_folder, 'k49-test-imgs.npz'))['arr_0']
            ),
            torch.tensor(
                np.load(os.path.join(self.raw_folder, 'k49-test-labels.npz'))['arr_0']
            )
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)


class EMNIST(TV_EMNIST):
    url = 'https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    splits = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')
    classes = {
        'byclass':
            [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                'y', 'z'
            ],
        'bymerge':
            [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q',
                'r', 't'
            ],
        'balanced':
            [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q',
                'r', 't'
            ],
        'letters':
            [
                'A, a', 'B, b', 'C, c', 'D, d', 'E, e', 'F, f', 'G, g', 'H, h', 'I, i',
                'J, j', 'K, k', 'L, l', 'M, m', 'N, n', 'O, o', 'P, p', 'Q, q', 'R, r',
                'S, s', 'T, t', 'U, u', 'V, v', 'W, w', 'X, x', 'Y, y', 'Z, z'
            ],
        'digits': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    }

    def __init__(self, root, split, **kwargs):
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        self.classes = self.classes[split]
        super(TV_EMNIST, self).__init__(root, **kwargs)


class Omniglot(VisionDataset):
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes_file = 'classes.pt'
    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found.' + ' You can use download=True to download it'
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )
        self.classes = torch.load(os.path.join(self.processed_folder, self.classes_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (
            os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
            os.path.exists(os.path.join(self.processed_folder, self.test_file))
        )

    def download(self):
        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        self._classes = None
        omni_back = TV_Omniglot(self.root, background=True, download=True)
        omni_fore = TV_Omniglot(self.root, background=False, download=True)

        print('Processing...')

        data = []
        targets = []
        for d, t in omni_back:
            data.append(np.array(ImageOps.invert(d))), targets.append(t)
        back_data, back_targets = torch.tensor(data), torch.tensor(targets)
        characters = omni_back._characters

        data = []
        targets = []
        for d, t in omni_fore:
            data.append(np.array(ImageOps.invert(d))), targets.append(len(characters) + t)
        fore_data, fore_targets = torch.tensor(data), torch.tensor(targets)
        characters += omni_fore._characters

        data = torch.cat([back_data, fore_data])
        targets = torch.cat([back_targets, fore_targets])

        train_idx = np.concatenate(
            [
                np.sort(np.random.choice(20, 17, replace=False)) + i
                for i in range(0,
                               len(data) - 1, 20)
            ]
        )
        test_idx = np.setdiff1d(np.arange(len(data)), train_idx)
        training_set = (data[train_idx], targets[train_idx])
        test_set = (data[test_idx], targets[test_idx])

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
        with open(os.path.join(self.processed_folder, self.classes_file), 'wb') as f:
            torch.save(characters, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
