import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
import os


class RandomGammaCorrection(object):
	def __init__(self, gamma = None):
		self.gamma = gamma
	def __call__(self,image):
		if self.gamma == None:
			# more chances of selecting 0 (original image)
			gammas = [0.5,1,2]
			self.gamma = random.choice(gammas)
			return TF.adjust_gamma(image, self.gamma, gain=1)
		elif isinstance(self.gamma,tuple):
			gamma=random.uniform(*self.gamma)
			return TF.adjust_gamma(image, gamma, gain=1)
		elif self.gamma == 0:
			return image
		else:
			return TF.adjust_gamma(image,self.gamma,gain=1)

def remove_background(image):
	#the input of the image is PIL.Image form with [H,W,C]
	image=np.float32(np.array(image))
	_EPS=1e-7
	rgb_max=np.max(image,(0,1))
	rgb_min=np.min(image,(0,1))
	image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
	image=torch.from_numpy(image)
	return image



class Blend_Image_Dataset(data.Dataset):
    def __init__(self, data_dir, transform_base=None, mode='valid'):
        
        self._load_blend_img_list(data_dir=data_dir)
        self._load_gt_img_list(data_dir=data_dir)
        self.transform_base=transform_base
    
    
        
    
    def _load_blend_img_list(self, data_dir):
        blend_dir = os.path.join(data_dir, "input")
        self.blend_list = [os.path.join(blend_dir, f) for f in os.listdir(blend_dir)]
        self.blend_list.sort()
    
    def _load_gt_img_list(self, data_dir):
        gt_dir = os.path.join(data_dir, "gt")
        self.gt_list = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]
        self.gt_list.sort()
    
    def __len__(self):
        return len(self.blend_list)
    
    def __getitem__(self, idx):
        blend_img_path = self.blend_list[idx]
        blend_img = Image.open(blend_img_path)
        
        gt_img_path = self.gt_list[idx]
        gt_img = Image.open(gt_img_path)
        
        to_tensor = transforms.ToTensor() 
        blend_img = to_tensor(blend_img)    
        gt_img = to_tensor(gt_img)
     
        
        return blend_img, gt_img
        
        
		
        
    
        

class Flare_Image_Dataset(data.Dataset):
	def __init__(self, image_path ,transform_base=None, mask_type=None ,mode='train'):
		assert mode in ['train', 'valid']
		self.mode = mode
		self.ext = ['png','jpeg','jpg','bmp','tif']
		self.data_list=[]
		[self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
		self.flare_dict={}
		self.flare_list=[]
		self.flare_name_list=[]

		self.reflective_flag=False
		self.reflective_dict={}
		self.reflective_list=[]
		self.reflective_name_list=[]

		self.lightsource_dict={}
		self.lightsource_list=[]
		self.lightsource_name_list=[]

		self.mask_type=mask_type #It is a str which may be None,"luminance" or "color"

		self.transform_base=transform_base


		print("Base Image Loaded with examples:", len(self.data_list))

	def transform_flare(self, flare, core):
		# angle, translate, scale, shear = transforms.RandomAffine.get_params(img_size=(1440,1440), degrees=(0,360), scale_ranges=(0.8, 1.5), translate=(300/1440, 300/1440), shears=(-20,20))
		# flare = TF.affine(flare, angle, translate, scale, shear)
		# core = TF.affine(core, angle, translate, scale, shear)

		centorcrop = transforms.CenterCrop((512,512))
		flare = centorcrop(flare)
		core = centorcrop(core)

		if random.random() > 0.5:
			flare = TF.vflip(flare)
			core = TF.vflip(core)

		if random.random() > 0.5:
			flare = TF.hflip(flare)
			core = TF.hflip(core)
		
		return flare, core

	def __getitem__(self, index):
		# load base image
		img_path=self.data_list[index]
		base_img= Image.open(img_path)
		
		gamma=np.random.uniform(1.8,2.2)
		to_tensor=transforms.ToTensor()
		adjust_gamma=RandomGammaCorrection(gamma)
		adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
		color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)
		if self.transform_base is not None:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
			base_img=self.transform_base(base_img)
		else:
			base_img=to_tensor(base_img)
			base_img=adjust_gamma(base_img)
			base_img=base_img.permute(2,0,1)
		sigma_chi=0.01*np.random.chisquare(df=1)
		base_img=Normal(base_img,sigma_chi).sample()
		gain=np.random.uniform(0.5,1.2)
		flare_DC_offset=np.random.uniform(-0.02,0.02)
		base_img=gain*base_img
		base_img=torch.clamp(base_img,min=0,max=1)

		#load flare image and light source
		flare_path=random.choice(self.flare_list)
		flare_idx=self.flare_list.index(flare_path)
		flare_img =Image.open(flare_path)
		if self.reflective_flag:
			reflective_path=random.choice(self.reflective_list)
			reflective_img =Image.open(reflective_path)

		lightsource_path=self.lightsource_list[flare_idx]
		lightsource_img =Image.open(lightsource_path)

		flare_img=to_tensor(flare_img)
		lightsource_img=to_tensor(lightsource_img)
  
  
		flare_img, lightsource_img = self.generate_multi_flare(flare_img, lightsource_img)    
		flare_img=adjust_gamma(flare_img)
  
            
		lightsource_img=adjust_gamma(lightsource_img)

		
		if self.reflective_flag:
			reflective_img=to_tensor(reflective_img)
			reflective_img=adjust_gamma(reflective_img)
			flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

		flare_img=remove_background(flare_img)
		lightsource_img=remove_background(lightsource_img)
		
		flare_img, lightsource_img = self.transform_flare(flare_img, lightsource_img)
		
		#change color
		flare_img=color_jitter(flare_img)
		lightsource_img=color_jitter(lightsource_img)

		#flare and light source blur
		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		flare_img=blur_transform(flare_img)
		flare_img=flare_img+flare_DC_offset
		flare_img=torch.clamp(flare_img,min=0,max=1)

		blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
		lightsource_img=blur_transform(lightsource_img)
		lightsource_img=lightsource_img+flare_DC_offset
		lightsource_img=torch.clamp(lightsource_img,min=0,max=1)

		#merge image	
		merge_img=flare_img+base_img
		merge_img=torch.clamp(merge_img,min=0,max=1)
		merge_lightsource_img=lightsource_img+base_img
		merge_lightsource_img=torch.clamp(merge_lightsource_img, min=0, max=1)


		if self.mask_type==None:
			return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),gamma,adjust_gamma_reverse(merge_lightsource_img)
		elif self.mask_type=="luminance":
			#calculate mask (the mask is 3 channel)
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			luminance=0.3*flare_img[0]+0.59*flare_img[1]+0.11*flare_img[2]
			threshold_value=0.99**gamma
			flare_mask=torch.where(luminance >threshold_value, one, zero)

		elif self.mask_type=="color":
			one = torch.ones_like(base_img)
			zero = torch.zeros_like(base_img)

			threshold_value=0.99**gamma
			flare_mask=torch.where(merge_img >threshold_value, one, zero)

		return adjust_gamma_reverse(base_img),adjust_gamma_reverse(flare_img),adjust_gamma_reverse(merge_img),flare_mask,gamma, adjust_gamma_reverse(merge_lightsource_img)

	def __len__(self):
		return len(self.data_list)
	
	def load_scattering_flare(self,flare_name,flare_path):
		flare_list=[]
		[flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
		self.flare_name_list.append(flare_name)
		self.flare_dict[flare_name]=flare_list
		self.flare_list.extend(flare_list)
		len_flare_list=len(self.flare_dict[flare_name])
		if len_flare_list == 0:
			print("ERROR: scattering flare images are not loaded properly")
		else:
			print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
		print("Now we have",len(self.flare_list),'scattering flare images')

	def load_reflective_flare(self,reflective_name,reflective_path):
		self.reflective_flag=True
		reflective_list=[]
		[reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
		self.reflective_name_list.append(reflective_name)
		self.reflective_dict[reflective_name]=reflective_list
		self.reflective_list.extend(reflective_list)
		len_reflective_list=len(self.reflective_dict[reflective_name])
		if len_reflective_list == 0:
			print("ERROR: reflective flare images are not loaded properly")
		else:
			print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
		print("Now we have",len(self.reflective_list),'refelctive flare images')

	def load_light_source(self,source_name,source_path):
		lightsource_list=[]
		[lightsource_list.extend(glob.glob(source_path + '/*.' + e)) for e in self.ext]
		self.lightsource_name_list.append(source_name)
		self.lightsource_dict[source_name]=lightsource_list
		self.lightsource_list.extend(lightsource_list)
		len_source_list=len(self.lightsource_dict[source_name])
		if len_source_list == 0:
			print("ERROR: light source images are not loaded properly")
		else:
			print("Light Source Image:",source_name, " is loaded successfully with examples", str(len_source_list))
		print("Now we have",len(self.lightsource_list),'light source images')
  

	def generate_multi_flare(self, flare_img, light_img):
		num = np.random.randint(8)
		mul_flare_img = torch.zeros_like(flare_img)
		mul_light_img = torch.zeros_like(light_img)
		
		if num >= 3:
			for i in range(num):
                
				angle, translate, scale, shear = transforms.RandomAffine.get_params(img_size=(1440,1440), degrees=(0,360),scale_ranges=(0.4, 1.0),translate=(300/1440,300/1440),shears=(-20,20))
				mul_flare_img += TF.affine(flare_img, angle, translate, scale, shear)
				mul_light_img += TF.affine(light_img, angle, translate, scale, shear)
				

		else:
			if num == 0:
				angle, translate, scale, shear = transforms.RandomAffine.get_params(img_size=(1440,1440), degrees=(0,360),scale_ranges=(0.8,1.5),translate=(300/1440,300/1440),shears=(-20,20))
				flare_img = TF.affine(flare_img, angle, translate, scale, shear)
				light_img = TF.affine(light_img, angle, translate, scale, shear)
				return flare_img, light_img	
			else:
				for i in range(num):
                    
					angle, translate, scale, shear = transforms.RandomAffine.get_params(img_size=(1440,1440), degrees=(0,360),scale_ranges=(0.8,1.5),translate=(300/1440,300/1440),shears=(-20,20))
					mul_flare_img += TF.affine(flare_img, angle, translate, scale, shear)           
					mul_light_img += TF.affine(light_img, angle, translate, scale, shear)
					

		
		

		return mul_flare_img, mul_light_img