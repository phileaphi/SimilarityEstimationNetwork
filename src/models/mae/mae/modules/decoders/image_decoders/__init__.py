__author__ = 'max'

from .binary_image_decoder import BinaryImageDecoder
from .color_image_decoder import ColorImageDecoder
from .resnet import ResnetDecoderBinaryImage28x28, ResnetDecoderColorImage32x32
from .pixelcnn import PixelCNNDecoderBinaryImage28x28
from .pixelcnnpp import PixelCNNPPDecoderColorImage32x32, PixelCNNPPDecoderColorImage64x64
