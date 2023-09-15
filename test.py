import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import PIL.Image as pil
import argparse
import glob
import os

import synthesis
from networks import unet



parser = argparse.ArgumentParser()
parser.add_argument('--ckp_path',
                    type=str, default='./pretrained/epoch_020.pt')
parser.add_argument('--image_path', type=str, default='./data/test')
parser.add_argument('--result_path', type=str, default='./data/result')
parser.add_argument('--ext', type=str, default="png")

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def save_outputs(result, result_path, idx, resolution=512):
    for k in ["input", "pred_blend", "pred_scene", "pred_flare"]:
        image = result[k]
        if max(image.shape[-1], image.shape[-2]) > resolution:
            image = T.functional.resize(image, [resolution, resolution], antialias=True)

        seperate_path = os.path.join(result_path, k)
        if not os.path.exists(seperate_path):
             os.makedirs(seperate_path)
        torchvision.utils.save_image(image, seperate_path+ '/' + str(idx) + '.' + args.ext)


def remove_flare(model, image):

    """
    process one image
    """

    inputs = image
    w, h = inputs.size

    inputs = T.ToTensor()(inputs)
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
        pred_blend=pred_blend.cpu(),
        pred_scene=pred_scene.cpu(),
        pred_flare=pred_flare.cpu()
    )


def test(args):

    # model 불러오기
    ckp_path = args.ckp_path
    print("Loading model from", ckp_path)
    model = unet.UNet(in_channels=3, out_channels=3).cuda()
    ckp = torch.load(ckp_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckp["g"])
    model.eval()

    # result path
    result_path = args.result_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # input image path
    paths = args.image_path
    if os.path.isdir(paths):
        # 이미지 여러개 테스트
        paths = glob.glob(os.path.join(paths, '*.{}'.format(args.ext)))
    elif os.path.isfile(paths):
        # 이미지 한 개 테스트
        paths = [paths]
    else:
        raise Exception("Can't find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # Flare removal 
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            
            # Load images
            inputs = pil.open(image_path).convert('RGB')
            results = remove_flare(model, inputs)
            save_outputs(results, result_path, idx)
            # test = T.ToPILImage()(results['pred_blend'].squeeze(0))
            # test.show()
            
    print("Done!!")


if __name__ == '__main__':
    test(args)