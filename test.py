<<<<<<< HEAD
from collections import defaultdict
from pathlib import Path
import os
import kornia as K
from networks import UNet
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional
from tqdm import tqdm
from options import DeflareOptions
from data_loader import  Blend_Image_Dataset
import synthesis
import utils
import argparse
import networks

device = torch.device("cpu" if torch.cuda.is_available else "cuda")

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--image_path', type=str, default='./data/test')
parser.add_argument('--output_path', type=str, default='./data/test_output')
parser.add_argument('--ckp_path', type = str, help = "Checkpoint path for model", default = './ckp/epoch_020.pt')

arg = parser.parse_args()

def save_images(result, output_path, idx, resolution=512):
=======
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict
from utils import get_logger
import kornia as K
import logging
import argparse
import glob
import os

import synthesis
from networks import unet
from data_loader import Blend_Image_Dataset



parser = argparse.ArgumentParser()
parser.add_argument('--ckp_path',
                    type=str, default='./pretrained/epoch_020.pt')
parser.add_argument('--image_path', type=str, default='./data/test')
parser.add_argument('--result_path', type=str, default='./data/result')
parser.add_argument('--ext', type=str, default="png")
parser.add_argument('--log_path', type=str, default='./log')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def save_outputs(result, result_path, idx, resolution=512):
>>>>>>> 19406855300e37c52b2a2b50f60920e6300b7800
    for k in ["input", "pred_blend", "pred_scene", "pred_flare"]:
        image = result[k]
        if max(image.shape[-1], image.shape[-2]) > resolution:
            image = T.functional.resize(image, [resolution, resolution], antialias=True)
<<<<<<< HEAD
            
        seperate_path = os.path.join(output_path, k)
        if not os.path.exists(output_path):
             os.makedirs(output_path)
    
    torchvision.utils.save_image(image, output_path + '/' + str(idx) + '.png')
    
def single_remove_flare(image, model, resolution=512, high_resolution=2048):
    input = image
    _, h, w = input.shape
    input = input.cuda().unsqueeze(0)

    if min(h, w) >= high_resolution:
        images = T.functional.center_crop(input, [high_resolution, high_resolution])
        images_low = F.interpolate(images, (resolution, resolution), mode="area")
        pred_img_low = model(images_low).clamp(0.0, 1.0)
        pred_flare_low = synthesis.remove_flare(images_low, pred_img_low)
        pred_flare = T.functional.resize(pred_flare_low, [high_resolution, high_resolution], antialias=True)
        pred_scene = synthesis.remove_flare(images, pred_flare)
    else:
        images = T.functional.center_crop(input, [resolution, resolution])
        pred_scene = model(images).clamp(0.0, 1.0)
        pred_flare = synthesis.remove_flare(images, pred_scene)

    pred_blend = synthesis.blend_light_source(images.cpu(), pred_scene.cpu())
    
    return dict(
        input=images.cpu(),
=======

        seperate_path = os.path.join(result_path, k)
        if not os.path.exists(seperate_path):
             os.makedirs(seperate_path)
        torchvision.utils.save_image(image, seperate_path+ '/' + str(idx) + '.' + args.ext)


def remove_flare(model, image):

    """
    process one image
    """

    inputs = image
    _, w, h = inputs.shape

    inputs = inputs.cuda().unsqueeze(0)     # (1,3,h,w)


    if min(h, w) >= 2048:
        inputs = T.functional.center_crop(inputs, [2048,2048])
        inputs_low = F.interpolate(inputs, (512,512), mode='area')
        pred_scene_low = model(inputs_low).clamp(0.0, 1.0)
        pred_flare_low = synthesis.remove_flare(inputs_low, pred_scene_low)
        pred_flare = T.functional.resize(pred_flare_low, [2048,2048], antialias=True)
        pred_scene = synthesis.remove_flare(inputs, pred_flare)
    else:
        inputs = T.functional.center_crop(inputs, [512,512])
        pred_scene = model(inputs).clamp(0.0, 1.0)
        pred_flare = synthesis.remove_flare(inputs, pred_scene)

    pred_blend = synthesis.blend_light_source(inputs.cpu(), pred_scene.cpu())

    return dict(
        input=inputs.cpu(),
>>>>>>> 19406855300e37c52b2a2b50f60920e6300b7800
        pred_blend=pred_blend.cpu(),
        pred_scene=pred_scene.cpu(),
        pred_flare=pred_flare.cpu()
    )

<<<<<<< HEAD
def load(filename, net, optim=None):
    dict_model = torch.load(filename)
    
    net.load_state_dict(dict_model['g'])
    if optim is not None:
        optim.load_state_dict(dict_model['optim'])

    return net, optim
      
def test(args):
    
    ckp_path = args.ckp_path
    
    test_dataset = Blend_Image_Dataset(args.image_path)
    
    model = UNet(in_channels=3, out_channels=3).cuda()
    model = load(ckp_path, model)[0]
    model.eval()
    
    metrics = defaultdict(float)
    
    with torch.no_grad():
        for idx, (image, gt) in enumerate(test_dataset):
            results = single_remove_flare(image, model)
            save_images(results, args.output_path, idx)
            
            metrics['PSNR'] = K.metrics.psnr(results['pred_blend'].squeeze(0), gt, 1.0).item()
            metrics['SSIM'] = K.metrics.ssim(results['pred_blend'], gt.unsqueeze(0), 11).mean().item()
    
    
if __name__ == '__main__':
    test(arg)   
=======

def test(args):

    # model 불러오기
    ckp_path = args.ckp_path

    if os.path.isfile(ckp_path):
        print("Loading model from", ckp_path)
        model = unet.UNet(in_channels=3, out_channels=3).cuda()
        ckp = torch.load(ckp_path, map_location=torch.device("cpu"))
        model.load_state_dict(ckp["g"])
        model.eval()
    else: raise Exception("Can't find args.ckp_path: {}".format(args.ckp_path))

    # result path
    result_path = args.result_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # logging
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_filename = os.path.join(args.log_path, 'test.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    metrics = defaultdict(float)

    # dataloader
    test_dataloader = Blend_Image_Dataset(args.image_path)

    with torch.no_grad():
        for idx, (image, gt) in enumerate(test_dataloader):
            results = remove_flare(model, image)
            save_outputs(results, result_path, idx)
            
            metrics['PSNR'] = K.metrics.psnr(results['pred_blend'].squeeze(0), gt, 1.0).item()
            metrics['SSIM'] = K.metrics.ssim(results['pred_blend'], gt.unsqueeze(0), 11).mean().item()

            logger.info(
                f"Test Image [{idx}/{len(test_dataloader)}] metrics "
                + "\t".join([f"{k}={v:.4f}" for k, v in metrics.items()]))


    print("Done!!")


if __name__ == '__main__':
    test(args)
>>>>>>> 19406855300e37c52b2a2b50f60920e6300b7800
