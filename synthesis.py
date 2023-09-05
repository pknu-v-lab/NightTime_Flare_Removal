import random
from typing import Union, Sequence

import kornia
import kornia.geometry.transform as KT
import numpy as np
import torch
import torch.nn.functional
import torchvision.transforms as T
import torchvision.transforms.functional
import torchvision.utils
import cv2
import skimage
import skimage.measure
import skimage.morphology

import utils

# Small number added to near-zero quantities to avoid numerical instability.
_EPS = 1e-7

def remove_flare(combined, flare, gamma=2.2):
    # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
    # gamma < 1.
    combined = combined.clamp(_EPS, 1.0)
    flare = flare.clamp(_EPS, 1.0)
    gamma = gamma.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    combined_linear = torch.pow(combined, gamma)
    flare_linear = torch.pow(flare, gamma)

    scene_linear = combined_linear - flare_linear
    # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
    # gamma < 1.
    scene_linear = scene_linear.clamp(_EPS, 1.0)
    scene = torch.pow(scene_linear, 1.0 / gamma)
    return scene

def get_highlight_mask(image, threshold=0.99):
    binary_mask = image.mean(dim=1, keepdim=True) > threshold
    binary_mask = binary_mask.to(image.dtype)
    return binary_mask