import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Grayscale(nn.Module):
    # http://poynton.ca/PDFs/ColorFAQ.pdf
    # 0.2125 R + 0.7154 G + 0.0721 B
    # BxCxHxW
    def __call__(self, im: torch.Tensor):
        if im.size(1) > 1:
            im = (
                im[:, 0, :, :] * 0.2125 + im[:, 1, :, :] * 0.7154 +
                im[:, 2, :, :] * 0.0721
            ).unsqueeze(1)
        return im


class BinarizeImage(nn.Module):
    def __call__(self, im: torch.Tensor):
        im = torch.rand(im.size()).type_as(im).le(im).float()
        return im


class Interpolate(nn.Module):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, im: torch.Tensor):
        if tuple(im.shape[-2:]) != self.target_size:
            im = F.interpolate(
                im, size=self.target_size, mode='bicubic', align_corners=False
            )
        return im


class Clamp(nn.Module):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, im: torch.Tensor):
        im = torch.clamp(im, min=self.min, max=self.max)
        return im


class ElasticTransform(nn.Module):
    def __init__(self, alpha, sigma, random_state=None):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma / 3
        self.random_state = None

    def __call__(self, image):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        nim = np.array(image)
        if self.random_state is None:
            self.random_state = np.random.RandomState(None)

        nim = np.expand_dims(nim, -1) if len(nim.shape) < 3 else nim
        shape = nim.shape
        dx = gaussian_filter(
            (self.random_state.rand(*shape) * 2 - 1),
            image.width * self.sigma,
            truncate=1
        ) * self.alpha
        dy = gaussian_filter(
            (self.random_state.rand(*shape) * 2 - 1),
            image.height * self.sigma,
            truncate=1
        ) * self.alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
        )
        indices = np.reshape(y + dy,
                             (-1, 1)), np.reshape(x + dx,
                                                  (-1, 1)), np.reshape(z, (-1, 1))

        distorted_image = map_coordinates(nim, indices, order=1,
                                          mode='reflect').reshape(nim.shape)
        im = Image.fromarray(distorted_image.squeeze())
        return im


class Normalize(nn.Module):
    '''
    Normalize from the 0-255 range to the 0-1 range.
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor):
        x = x / 255.0
        return x


class ToNumpy(nn.Module):
    '''
    Convert PIL image to a numpy array
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, x: np.array):
        x = np.expand_dims(np.array(x), 0)
        return x


class NONE(transforms.Compose):
    def __init__(self, *args, **kwargs):
        super().__init__([ToNumpy()])


class OCR(transforms.Compose):
    def __init__(
        self, train, resize_target, elastic_coeffs=(1., 1.), degrees=0., dist_scale=0.
    ):
        if train:
            super().__init__(
                [
                    ElasticTransform(*elastic_coeffs),
                    transforms.RandomAffine(degrees),
                    transforms.RandomPerspective(dist_scale),
                    ToNumpy(),
                ]
            )
        else:
            super().__init__([
                ToNumpy(),
            ])
