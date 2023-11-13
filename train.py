from __future__ import absolute_import, division, print_function

from time import time
import numpy as np
import random
import torch
import kornia as K
from collections import defaultdict
from utils import get_logger, build_criterion,grid_transpose, log_time, load_ckp
from torch.optim import Adam, lr_scheduler
import torch.backends.cudnn as cudnn
from networks import *
import synthesis
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_loader import Flare_Image_Dataset, Blend_Image_Dataset
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter
from options import DeflareOptions
import os
import json
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
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        
        self.running_scalars = defaultdict(float)
        self.logger = get_logger(self)
        self.writers = {}

        self.tb_writers = SummaryWriter(self.log_path,self.opt.model_name)
        
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
        
        self.val_flare_image_dataset = Blend_Image_Dataset(self.opt.val, transform_base=self.transform_base, mode='valid')
        self.val_dataloader = DataLoader(dataset=self.val_flare_image_dataset,
                                         batch_size=1,
                                         shuffle=False)
            
        
        num_train_samples = len(self.train_dataloader)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epoch
        
        
        if self.opt.model == 'NAFNet':
            self.model = NAFNet().to(self.device)
        
        if self.opt.model == 'UNet':
            self.model = UNet(in_channels=3, out_channels=3).to(self.device)
        
        self.init_weights(self.model)
                                    
        #optimizer
        self.optimizer = Adam(self.model.parameters(), self.opt.lr)
        self.scheduler = lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[25, 35], gamma=0.5)
        torch.optim.lr_scheduler.MultiStepLR
        
        self.criterion = build_criterion(self)
        
        self.save_opts()
        
        
    def init_weights(self, net, init_type="xavier_uniform", init_gain=1):


        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == "xavier_normal":
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == "kaiming":
                    init.kaiming_normal_(
                        m.weight.data, a=0, mode="fan_in", nonlinearity="relu"
                    )
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif (
                classname.find("BatchNorm2d") != -1
            ):  # BatchNorm Layer's weight is not a matrix;
                # only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        self.logger.info("initialize network with %s" % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>
        
    def set_train(self):
        """Convert all models to training mode
        """
        self.model.train()
            
    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.model.eval()
            
    def train(self):
        """Run the entire training pipeline
        """

        self.epoch = 0
        self.step = 0
        self.start_time = time()

        if self.opt.resume == True:
            ckp_list = os.listdir(self.output_dir)
            ckp_list.sort()
            self.model, self.optimizer, self.epoch = load_ckp(
                os.path.join(self.output_dir, ckp_list[-1]),
                self.model, self.optimizer)
            print(f">> Training starts from the Epoch {self.epoch + 1}")

            for self.epoch in range(self.epoch + 1, self.opt.num_epoch):
                self.run_epoch()

        else:
            for self.epoch in range(self.opt.num_epoch):
                self.run_epoch()

        self.logger.success(f"train over.")
            
     
    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        self.scheduler.step()


        print("Training")
        self.set_train()
        
        for batch_idx, inputs in enumerate(self.train_dataloader):
            
            before_op_time = time()
            scene_img, flare_img, merge_img, gamma = inputs
            
            scene_img = scene_img.to(self.device).float()
            flare_img = flare_img.to(self.device).float()
            merge_img = merge_img.to(self.device).float()
            gamma = gamma.to(self.device).float()
            

            pred_scene = self.model(merge_img).clamp(0.0, 1.0)
            pred_flare = synthesis.remove_flare(merge_img, pred_scene, gamma)
            
            flare_mask = synthesis.get_highlight_mask(flare_img)
            
            # Fill the saturation region with the ground truth, so that no L1/L2
            # loss and better for perceptual loss since
            # it matches the surrounding scenes.
            masked_scene = pred_scene * (1 - flare_mask) + scene_img * flare_mask
            masked_flare = pred_flare * (1 - flare_mask) + flare_img * flare_mask
            
            loss = dict()
            loss_weights = self.opt.loss_weight
            
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
        
            duration = time() - before_op_time
            ########################
            early_phase = batch_idx % self.opt.log_frequency == 0 
            late_phase = self.step % 2000 == 0
            
            
            
            if early_phase or late_phase:
                log_time(self, batch_idx, duration, total_loss.cpu())
                
                self.val()
                
                
                
                
            for k, v in loss.items():
                self.running_scalars[k] = self.running_scalars[k] + v.detach().mean().item()
                
            global_step = self.step
            
            if global_step % 500 == 0:
                self.tb_writers.add_scalar(
                    "metric/total_loss", total_loss.detach().cpu().item(), global_step
                )
                for k in self.running_scalars:
                    v = self.running_scalars[k] / 100
                    self.running_scalars[k] = 0.0
                    self.tb_writers.add_scalar(f"loss/{k}", v, global_step)

            if global_step % 500 == 0:
                images = grid_transpose(
                    [merge_img, scene_img, pred_scene, flare_img, pred_flare]
                )
                images = torchvision.utils.make_grid(
                    images, nrow=5, value_range=(0, 1), normalize=True
                )
                self.tb_writers.add_image(
                    f"train/combined|real_scene|pred_scene|real_flare|pred_flare",
                    images,
                    global_step,
                )
            self.step += 1
            
        self.logger.info(
            f"EPOCH[{self.epoch}/{self.opt.num_epoch}] END "
            f"Taken {(time() - before_op_time) / 60.0} min"
        )
        
        if self.epoch % 2 == 0:
            to_save = dict(
                g=self.model.state_dict(), g_optim=self.optimizer.state_dict(), epoch=self.epoch
            )
            torch.save(to_save, os.path.join(self.output_dir, f"epoch_{self.epoch:03d}.pt"))
        
    
        
    def val(self):
        
        with torch.no_grad():
                
                    self.model.eval()

                    metrics = defaultdict(float)
                    for n, (inputs, labels) in enumerate(self.val_dataloader):
                        inputs = inputs.to(self.device)
                        

                        results = synthesis.batch_remove_flare(self, inputs, self.model, resolution=512)
                        metrics["PSNR"] += K.metrics.psnr(results["pred_blend"], labels, 1.0).item()
                        metrics["SSIM"] += (
                            K.metrics.ssim(results["pred_blend"], labels, 11).mean().item()
                        )

                    for k in metrics:
                        metrics[k] = metrics[k] / len(self.val_dataloader)

                    self.model.train()

                    self.logger.info(
                        f"EPOCH[{self.epoch}/{self.opt.num_epoch}] metrics "
                        + "\t".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

                    for m, v in metrics.items():
                        self.tb_writers.add_scalar(f"evaluate/{m}", v, self.epoch * len(self.train_dataloader))
    
    def save_opts(self):
        
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=4)

       
                
                    
                
            
  




if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
    
