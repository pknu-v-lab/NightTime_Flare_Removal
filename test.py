from collections import defaultdict
import os
import kornia as K
from networks import UNet
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional
import logging

from data_loader import  Blend_Image_Dataset
import synthesis
import argparse
from networks import UNet

device = torch.device("cpu" if torch.cuda.is_available else "cuda")

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--image_path', type=str, default='./data/test')
parser.add_argument('--result_path', type=str, default='./data/result')
parser.add_argument('--ckp_path', type = str, help = "Checkpoint path for model", default = './checkpoints/epoch_044.pt')
parser.add_argument('--log_path', type = str, help = "Path to log", default = './log' )
arg = parser.parse_args()


def save_images(result, output_path, idx, resolution=512):
    for k in ["input", "pred_blend", "pred_scene", "pred_flare"]:
        image = result[k]
        if max(image.shape[-1], image.shape[-2]) > resolution:
            image = T.functional.resize(image, [resolution, resolution], antialias=True)
            
        seperate_path = os.path.join(output_path, k)
        if not os.path.exists(seperate_path):
             os.makedirs(seperate_path)
        torchvision.utils.save_image(image, seperate_path + '/' + str(idx) + '.png')
    
def single_remove_flare(model, image, resolution=512, high_resolution=2048):
    input = image
    _, h, w = input.shape
    input = input.cuda().unsqueeze(0)

    if min(h, w) >= high_resolution:
        input = T.functional.center_crop(input, [high_resolution, high_resolution])
        images_low = F.interpolate(input, (resolution, resolution), mode="area")
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
        input=images.cpu(), pred_blend=pred_blend.cpu(), pred_scene=pred_scene.cpu(), pred_flare=pred_flare.cpu()
    )

def load(filename, net, optim=None):
    dict_model = torch.load(filename)
    
    net.load_state_dict(dict_model['g'])
    if optim is not None:
        optim.load_state_dict(dict_model['optim'])

    return net, optim
      
def get_logger():
        if not os.path.exists('./log'):
            os.makedirs('./log')

        if not os.path.exists(arg.log_path):
            os.makedirs(arg.log_path)

        # log 출력 형식
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # log 출력
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # log를 파일에 출력
        log_filename = os.path.join(arg.log_path, 'test.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

def test(args):
    
    ckp_path = args.ckp_path
    
    model = UNet(in_channels=3, out_channels=3).cuda()
    model = load(ckp_path, model)[0]
    model.eval()
    
    result_path = args.result_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    logger = get_logger()
                                  
    metrics = defaultdict(float)
    
    test_dataset = Blend_Image_Dataset(args.image_path)
    
    with torch.no_grad():
        for idx, (image, gt) in enumerate(test_dataset):
            results = single_remove_flare(model, image)
            save_images(results, args.result_path, idx)
            
            metrics['PSNR'] = K.metrics.psnr(results['pred_blend'].squeeze(0), gt, 1.0).item()
            metrics['SSIM'] = K.metrics.ssim(results['pred_blend'], gt.unsqueeze(0), 11).mean().item()
    
            logger.info(
                f"Test Image [{idx}/{len(test_dataset)}] metrics " + "\t".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

    
if __name__ == '__main__':
    test(arg)   