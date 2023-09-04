from __future__ import absolute_import, division, print_function

from time import time
import logging
import numpy as np
import random
import torch
from torch.optim import Adam, lr_scheduler
import torch.backends.cudnn as cudnn
from networks import UNet
import torchvision.transforms as transforms
from data_loader import Flare_Image_Loader
from options import DeflareOptions
import os


options = DeflareOptions()
opts = options.parse()

# fix seed
random.seed(opts.seed)
np.random.seed(opts.seed)
torch.manual_seed(opts.seed)
torch.cuda.manual_seed_all(opts.seed)



class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        
        
        self.parameters_to_train = []
        
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        if self.device == 'cuda':
            if self.opt.deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                cudnn.benchmark = True
                
        self.output_dir = self.opt.output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
        self.transform_base=transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect'),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])

        self.transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
                              transforms.CenterCrop((512,512)),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])
        self.train_flare_image_loader = Flare_Image_Loader(self.opt.base_img, self.transform_base, self.transform_flare, mode='train')
        self.train_flare_image_loader.load_scattering_flare(self.opt.flare_img, os.path.join(self.opt.flare_img, 'Flare'))
                                                            
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)
                                    
        #optimizer
        self.optimizer = Adam(self.model.parameters(), self.opt.learning_rate)
        self.scheduler = lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[25, 35], gamma=0.5)
        torch.optim.lr_scheduler.MultiStepLR
        
        self.criterion = build
        
    
    def get_logger(self):
        if not os.path.exists('./log'):
            os.makedirs('./log')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # log 출력 형식
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # log 출력
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # log를 파일에 출력
        log_filename = os.path.join(log_dir, 'training.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def build_criterion(self):
        loss_weights = config.loss.weight

        criterion = dict()

        def _empty_l(*args, **kwargs):
            return 0

        def valid_l(name):
            return (
                loss_weights["flare"].get(name, 0.0) > 0
                or loss_weights["scene"].get(name, 0.0) > 0
            )

        criterion["l1"] = torch.nn.L1Loss().to(device) if valid_l("l1") else _empty_l
        criterion["lpips"] = LPIPSLoss().to(device) if valid_l("lpips") else _empty_l
        criterion["ffl"] = FocalFrequencyLoss().to(device) if valid_l("ffl") else _empty_l
        criterion["perceptual"] = (
            PerceptualLoss(**config.loss.perceptual).to(device)
            if valid_l("perceptual")
            else _empty_l
        )

    return criterion
  






data_list = train_flare_image_loader.data_list
print(len(data_list))


for i in range(len(data_list)):
    base_img, flare_img, merge_img, flare_mask = train_flare_image_loader[i]
    
    