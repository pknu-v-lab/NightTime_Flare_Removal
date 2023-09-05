from __future__ import absolute_import, division, print_function

from time import time
import numpy as np
import random
import torch
from collections import defaultdict
from utils import get_logger, build_criterion,grid_transpose
from torch.optim import Adam, lr_scheduler
import torch.backends.cudnn as cudnn
from networks import UNet
import synthesis
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loader import Flare_Image_Dataset, Blend_Image_Dataset
from torch.utils.tensorboard import SummaryWriter
from options import DeflareOptions
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



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
        self.log_path = os.path.join(self.opt.log_dir, "UNet")
        
        self.running_scalars = defaultdict(float)
        self.logger = get_logger(self)
        self.writers = {}

        self.tb_writers = SummaryWriter(self.log_path,"UNet")
        
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        if self.device == 'cuda':
            if self.opt.deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                cudnn.benchmark = True
                
        self.output_dir = self.opt.output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
        self.transform_base=transforms.Compose([transforms.RandomCrop((512,512),pad_if_needed=True,padding_mode='reflect'),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])

        self.transform_flare=transforms.Compose([transforms.RandomAffine(degrees=(0,360),scale=(0.8,1.5),translate=(300/1440,300/1440),shear=(-20,20)),
                              transforms.CenterCrop((512,512)),
							  transforms.RandomHorizontalFlip(),
							  transforms.RandomVerticalFlip()
                              ])
        self.train_flare_image_dataset = Flare_Image_Dataset(self.opt.base_img, self.transform_base, self.transform_flare, mode='train')
        self.train_flare_image_dataset.load_scattering_flare(self.opt.flare_img, os.path.join(self.opt.flare_img, 'Flare'))
        
        
        self.train_dataloader = DataLoader(dataset=self.train_flare_image_dataset,
                              batch_size=self.opt.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0)
        
        self.val_flare_image_dataset = Blend_Image_Dataset(self.opt, self.opt.val, transform_base=self.transform_base, mode='valid')
        self.val_dataloader = DataLoader(dataset=self.val_flare_image_dataset,
                                         batch_size=1,
                                         shuffle=False)
        
        
        
        
        
                                                            
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)
                                    
        #optimizer
        self.optimizer = Adam(self.model.parameters(), self.opt.learning_rate)
        self.scheduler = lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[25, 35], gamma=0.5)
        torch.optim.lr_scheduler.MultiStepLR
        
        self.criterion = build_criterion(self)
        
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
            
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
            
    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epoch):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

        
    
    
    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        
        self.model_lr_scheduler.step()


        print("Training")
        self.set_train()
        
        for batch_idx, inputs in enumerate(self.train_dataloader):
            
            before_op_time = time.time()
            inputs = inputs.to(self.device)
            scene_img, flare_img, merge_img, gamma = inputs
            
            pred_scene = self.model(merge_img).clamp(0.0, 1.0)
            pred_flare = synthesis.remove_flare(merge_img, pred_scene, gamma)
            
            flare_mask = synthesis.get_highlight_mask(flare_img)
            
            # Fill the saturation region with the ground truth, so that no L1/L2
            # loss and better for perceptual loss since
            # it matches the surrounding scenes.
            masked_scene = pred_scene * (1 - flare_mask) + scene_img * flare_mask
            masked_flare = pred_flare * (1 - flare_mask) + flare_img * flare_mask
            
            loss = dict()
            loss_weights = { 'flare': {'l1': 1, 'perceptual': 1}, 'scene': {'l1': 1,'perceptual': 1}}
            
            for t, pred, gt in[
                ("scene", masked_scene, scene_img),
                ("flare", masked_flare, flare_img)
            ]:
                l = dict(
                    l1 = self.criterion["l1"](pred, gt),
                    ffl = self.criterion["ffl"](pred, gt),
                    lpips = self.criterion["lpips"](pred, gt, value_range=(0, 1)),
                    perceptual = self.criterion["perceptual"](pred, gt, value_range=(0, 1)),
                )
                for k in l:
                    if loss_weights[t].get(k, 0.0) > 0:
                        loss[f"{t}_{k}"] = loss_weights[t].get(k, 0.0) * l[k]
            
            self.optimizer.zero_grad()
            total_loss = sum(loss.values())
            total_loss.backward()
            self.optimizer.step()
            
            ########################
      
            
            

            for k, v in loss.items():
                self.running_scalars[k] = self.running_scalars[k] + v.detach().mean().item()
                
            global_step = (self.epoch - 1) * len(self.train_flare_image_loader)
            
            if global_step % 100 == 0:
                self.tb_writer.add_scalar(
                    "metric/total_loss", total_loss.detach().cpu().item(), global_step
                )
                for k in self.running_scalars:
                    v = self.running_scalars[k] / 100
                    self.running_scalars[k] = 0.0
                    self.tb_writer.add_scalar(f"loss/{k}", v, global_step)

            if global_step % 200 == 0:
                images = grid_transpose(
                    [merge_img, scene_img, pred_scene, flare_img, pred_flare]
                )
                images = torchvision.utils.make_grid(
                    images, nrow=5, value_range=(0, 1), normalize=True
                )
                self.tb_writer.add_image(
                    f"train/combined|real_scene|pred_scene|real_flare|pred_flare",
                    images,
                    global_step,
                )
        self.logger.info(
            f"EPOCH[{self.epoch}/{self.opt.num_epoch}] END "
            f"Taken {(time.time() - before_op_time) / 60.0:.4f} min"
        )
        
        if self.epoch % 2 == 0:
            to_save = dict(
                g=self.model.state_dict(), g_optim=self.optimizer.state_dict(), epoch=self.epoch
            )
            torch.save(to_save, self.output_dir / f"epoch_{self.epoch:03d}.pt")
            self.logger.info(f"save checkpoint at {self.output_dir / f'epoch_{self.epoch:03d}.pt'}")
            
        if self.epoch % 1 == 0:
            self.model.eval()
            # with torch.no_grad():
                
            
  




if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
    