import random
from typing import Union, Sequence

import kornia
import kornia.geometry.transform as KT
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
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
    if torch.is_tensor(gamma):
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

def _create_disk_kernel(kernel_size):
    x = np.arange(kernel_size) - (kernel_size - 1) / 2
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    kernel = np.float32(rr <= np.max(x)) + _EPS
    kernel = kernel / np.sum(kernel)
    return kernel


def blend_light_source(input_scene, pred_scene):
    binary_mask = (get_highlight_mask(input_scene) > 0.5).to("cpu", torch.bool)
    binary_mask = binary_mask.squeeze(dim=1)  # (b, h, w)
    binary_mask = binary_mask.numpy()

    labeled = skimage.measure.label(binary_mask)
    properties = skimage.measure.regionprops(labeled)
    max_diameter = 0
    for p in properties:
        # The diameter of a circle with the same area as the region.
        max_diameter = max(max_diameter, p["equivalent_diameter"])

    mask = np.float32(binary_mask)

    kernel_size = round(1.5 * max_diameter)
    if kernel_size > 0:
        kernel = _create_disk_kernel(kernel_size)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask * 3.0, 0.0, 1.0)
        mask_rgb = np.stack([mask] * 3, axis=1)

        mask_rgb = torch.from_numpy(mask_rgb).to(input_scene.device, torch.float32)
        blend = input_scene * mask_rgb + pred_scene * (1 - mask_rgb)
    else:
        blend = pred_scene
    return blend


def batch_remove_flare(
    self,
    images,
    model,
    resolution=512,
    high_resolution=2048,
):
    _, _, h, w = images.shape

    if min(h, w) >= high_resolution:
        images = T.functional.center_crop(images, [high_resolution, high_resolution])
        images_low = F.interpolate(images, (resolution, resolution), mode="area")
        pred_img_low = model(images_low).clamp(0.0, 1.0)
        pred_flare_low = remove_flare(images_low, pred_img_low)
        pred_flare = T.functional.resize(
            pred_flare_low, [high_resolution, high_resolution], antialias=True
        )
        pred_scene = remove_flare(images, pred_flare)
    else:
        images = T.functional.center_crop(images, [resolution, resolution])
        pred_scene = model(images).clamp(0.0, 1.0)
        pred_flare = remove_flare(images, pred_scene)

    try:
        pred_blend = blend_light_source(images.cpu(), pred_scene.cpu())
    except cv2.error as e:
        self.logger.error(e)
        pred_blend = pred_scene
    return dict(
        input=images.cpu(),
        pred_blend=pred_blend.cpu(),
        pred_scene=pred_scene.cpu(),
        pred_flare=pred_flare.cpu(),
    )
    
    