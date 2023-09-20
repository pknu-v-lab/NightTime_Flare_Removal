from __future__ import absolute_import, division, print_function

from time import time
import numpy as np
import random
import torch
import kornia as K
from collections import defaultdict
from utils import get_logger, build_criterion,grid_transpose, log_time
from torch.optim import Adam, lr_scheduler
import torch.backends.cudnn as cudnn
from networks import UNet
import synthesis
from synthesis import remove_flare, blend_light_source
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loader import Flare_Image_Dataset, Blend_Image_Dataset
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
from options import DeflareOptions
import os
import torchvision.transforms as T
import logging
import cv2



options = DeflareOptions()
opts = options.parse()

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    if opts.deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


# result path
result_path = opts.results_dir

if not os.path.exists(result_path):
    os.makedirs(result_path)

os.makedirs(result_path+ "/input")
os.makedirs(result_path+ "/blend")
os.makedirs(result_path+ "/scene")
os.makedirs(result_path+ "/flare")


test_dataset = Blend_Image_Dataset(opts, opts.test_dir, mode='test')

test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False)

def load(filename, net, optim=None):
    dict_model = torch.load(filename)
    
    net.load_state_dict(dict_model['g'])
    if optim is not None:
        optim.load_state_dict(dict_model['g_optim'])

    return net, optim

def get_logger():
        # log 출력 형식
        logger = logging.getLogger()
        
        if len(logger.handlers) > 0:
            return logger
    
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
    

        # log 출력
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # log를 파일에 출력
        log_filename = os.path.join(opts.log_dir, 'test.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
#removal
def batch_remove_flare(
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
        logger.error(e)
        pred_blend = pred_scene
    return dict(
        input=images.cpu(),
        pred_blend=pred_blend.cpu(),
        pred_scene=pred_scene.cpu(),
        pred_flare=pred_flare.cpu(),
    )
    
    
    

#model
model = UNet(in_channels=3, out_channels=3).to(device)
model = load(opts.output_path, model)[0]
model.eval()
# model = torch.nn.DataParallel(model)

#inv_normalize_bayer = InverseNormalize(
#    (158.1977, 200.9817, 152.5043), 
#    (143.5275, 188.4728, 142.1568)).to(device)

# 결과 생성
with torch.no_grad():
    loss_arr = []
    psnr_arr = []
    
    for n, (inputs, labels) in enumerate(test_dataloader):
        
        inputs = inputs.to(device)
                        

        results = batch_remove_flare(images= inputs, model= model)
        
        torchvision.utils.save_image(results["input"],result_path+"/input/"+str(n).zfill(5)+"_input.png")
        torchvision.utils.save_image(results["pred_blend"],result_path+"/blend/"+str(n).zfill(5)+"_blend.png")
        torchvision.utils.save_image(results["pred_scene"],result_path+"/scene/"+str(n).zfill(5)+"_scene.png")
        torchvision.utils.save_image(results["pred_flare"],result_path+"/flare/"+str(n).zfill(5)+"_flare.png")
        
        
        
        metrics = defaultdict(float)
        metrics["PSNR"] += K.metrics.psnr(results["pred_blend"], labels, 1.0).item()
        metrics["SSIM"] += (
                            K.metrics.ssim(results["pred_blend"], labels, 11).mean().item())
        
        logger = get_logger()
        
        logger.info([f"{k}={v:.4f}" for k, v in metrics.items()])

       
       
       
     
        
       
    
    
    
    
    
