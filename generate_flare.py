import torchvision.transforms as transforms
from data_loader import Flare_Image_Loader
import argparse
import os

parser = argparse.ArgumentParser(description="Generate Flare-corrupted Image")
parser.add_argument('--output_dir', default="./data/Flickr24K/")
parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, help='dir of checkpoint')


args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


transform_base=transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect'),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])

transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
                              transforms.CenterCrop((512,512)),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])

train_flare_image_loader = Flare_Image_Loader('./data/train/Flickr24K', transform_base, transform_flare, mode='train')
train_flare_image_loader.load_scattering_flare('./data/train/flare','./data/train/flare/Flare')

data_list = train_flare_image_loader.data_list
print(len(data_list))


for i in range(len(data_list)):
    base_img, flare_img, merge_img, flare_mask = train_flare_image_loader[i]
    
    