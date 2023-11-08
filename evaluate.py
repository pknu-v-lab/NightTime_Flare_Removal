import os
from skimage.metrics import mean_squared_error
from skimage import io
import numpy as np
from statistics import mean
import sys

input_dir = './data/result'
output_dir = './data/result'

input_folder = os.path.join(input_dir, 'input')
gt_folder = os.path.join(input_dir, 'pred_blend') #pred_scene ? pred_blend?
mask_folder = os.path.join(input_dir, 'pred_flare')

output_filename = os.path.join(output_dir, 'scores.txt')

mask_type_list = ['glare','streak','global']
gt_list = sorted(os.listdir(gt_folder))

input_list = [input_image for input_image in os.listdir(input_folder)]
#input_list = list(map(lambda x: os.path.join(input_folder,x.replace('gt', 'input')), gt_list))

mask_list = list([mask_image for mask_image in os.listdir(mask_folder)])
#mask_list = list(map(lambda x: os.path.join(mask_folder,x.replace('gt', 'mask')), gt_list))

gt_list = list([gt_image for gt_image in os.listdir(gt_folder)])
#gt_list = list(map(lambda x: os.path.join(gt_folder,x), gt_list))

img_num = len(gt_list)
metric_dict = {'glare':[],'streak':[],'global':[]}

def extract_mask(img_seg):
    # Return a dict with 3 masks including streak,glare,global(whole image w/o light source), masks are returned in 3ch. 
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
    mask_dict={}
    streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
    glare_mask=(img_seg[:,:,1])/255
    global_mask=(255-img_seg[:,:,2])/255
    mask_dict['glare']=[np.sum(glare_mask)/(512*512),np.expand_dims(glare_mask,2).repeat(3,axis=2)] #area, mask
    mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)] 
    mask_dict['global']=[np.sum(global_mask)/(512*512),np.expand_dims(global_mask,2).repeat(3,axis=2)] 
    return mask_dict

for i in range(img_num):
    img_gt=io.imread(os.path.join(gt_folder,gt_list[i]))
    img_input=io.imread(os.path.join(input_folder,input_list[i]))
    img_seg=io.imread(os.path.join(mask_folder,mask_list[i]))
    for mask_type in mask_type_list:
        mask_area,img_mask=extract_mask(img_seg)[mask_type]
        if mask_area>0:
            img_gt_masked=img_gt*img_mask
            img_input_masked=img_input*img_mask
            input_mse=mean_squared_error(img_gt_masked, img_input_masked)/(255*255*mask_area)
            input_psnr=10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type].append(input_psnr)

glare_psnr=mean(metric_dict['glare'])
streak_psnr=mean(metric_dict['streak'])
global_psnr=mean(metric_dict['global'])

mean_psnr=mean([glare_psnr,streak_psnr,global_psnr])

with open(output_filename, 'w') as f:
    f.write('{}: {}\n'.format('G-PSNR', glare_psnr))
    f.write('{}: {}\n'.format('S-PSNR', streak_psnr))
    f.write('{}:{}\n'.format('global',global_psnr))
    f.write('{}: {}\n'.format('Score', mean_psnr))
    f.write('DEVICE: CPU\n')