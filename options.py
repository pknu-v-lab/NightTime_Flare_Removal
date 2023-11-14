from __future__ import absolute_import, division, print_function

import os 
import argparse

file_dir = os.path.dirname(__file__)    # the directory that options.py 

class DeflareOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="DeFlare Image")
        
        # PATHS
        self.parser.add_argument("--output_dir", 
                            type=str,
                            help="path to save the output data",
                            default="./output")
        self.parser.add_argument("--val", 
                            type=str,
                            help="path to save the output data",
                            default="./data/val")
        self.parser.add_argument("--base_img", 
                            type=str,
                            help="path to the training data with background",
                            default="./data/train/Flickr24K")
        self.parser.add_argument("--flare_img", 
                            type=str,
                            help="path to the training data with flare",
                            default="./data/train/flare")
        self.parser.add_argument("--ckpt_dir", 
                            default="./checkpoints", 
                            type=str, 
                            help="dir of checkpoint")
        self.parser.add_argument("--log_dir",
                                 default="./log",
                                 type=str,
                                 help="log directory")
        
        
        
        # Training options
        self.parser.add_argument("--model_name", 
                                 type=str, 
                                 help="the name of the folder to save the model",
                                 default="NAFNet_Multi")
        self.parser.add_argument("--model",
                                 type=str,
                                 default="NAFNet",
                                 help="available options: NAFNet, UNet, UFormer")
        self.parser.add_argument('--deterninistic', 
                                 default=False, 
                                 type=bool,
                                 help='reproduce by fixing random seed')
        self.parser.add_argument('--seed', 
                                 default=2023, 
                                 type=int, 
                                 help='random crop size')
        self.parser.add_argument("--resume",
                                 default=False,
                                 type=bool,
                                 help="resume training from checkpoint")
        
        
        # Optimization options 
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=2)
        self.parser.add_argument("--lr",
                                 default=5e-5,
                                 type=float,
                                 help="learning rate")
        self.parser.add_argument("--num_epoch", 
                                 default=100, 
                                 type=int, 
                                 help="total epoch")
        
        self.parser.add_argument('--weight_decay', 
                                 type=float, 
                                 default=0.02, 
                                 help='weight decay')
        
        self.parser.add_argument('--iterations', 
                                 type=int, 
                                 default=600000, 
                                 help='max iterations')
        
        
        # System options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)
        
        
        # Loading Options
        # self.parser.add_argument("load_weights_folder",
        #                          type=str,
        #                          help="name of model to load")
        
        
        
        # Logging options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=200)
        
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        
        
        
        
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options