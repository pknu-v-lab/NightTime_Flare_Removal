import torch
from typing import Iterable, Union, Optional, MutableMapping
from losses.focal_frequency_loss import FocalFrequencyLoss
from losses.lpips_loss import LPIPSLoss
from losses.perceptual_loss import PerceptualLoss
import logging
import os
import time

def build_criterion(self):
        loss_weights = { 'flare': {'l1': 0, 'perceptual': 0, 'lpips' : 1, 'ffl':100},
                        'scene': {'l1': 1,'perceptual': 0, 'lpips' : 1, 'ffl' : 0}}

        
        loss_perceptual = {'layers': {'conv1_2': 0.384615,
                                      'conv2_2': 0.208333,
                                      'conv3_2': 0.270270,
                                      'conv4_2': 0.178571,
                                      'conv5_2': 6.666666},
                           'criterion': 'L1'}
                          
        

        criterion = dict()

        def _empty_l(*args, **kwargs):
            return 0

        def valid_l(name):
            return (
                loss_weights["flare"].get(name, 0.0) > 0
                or loss_weights["scene"].get(name, 0.0) > 0
            )

        criterion["l1"] = torch.nn.L1Loss().to(self.device) if valid_l("l1") else _empty_l
        criterion["lpips"] = LPIPSLoss().to(self.device) if valid_l("lpips") else _empty_l
        criterion["ffl"] = FocalFrequencyLoss().to(self.device) if valid_l("ffl") else _empty_l
        criterion["perceptual"] = (
            PerceptualLoss(**loss_perceptual).to(self.device)
            if valid_l("perceptual")
            else _empty_l
        )

        return criterion
    
def get_logger(self):
        if not os.path.exists('./log'):
            os.makedirs('./log')

        if not os.path.exists(self.opt.log_dir):
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
        log_filename = os.path.join(self.opt.log_dir, 'training.log')
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    

def load_ckp(checkpoint_path, model, optimizer):
     checkpoint = torch.load(checkpoint_path)
     model.load_state_dict(checkpoint["g"])
     optimizer.load_state_dict(checkpoint["g_optim"])

     return model, optimizer, checkpoint['epoch']


def grid_transpose(
    tensors: Union[torch.Tensor, Iterable], original_nrow: Optional[int] = None
) -> torch.Tensor:
    """
    batch tensors transpose.
    :param tensors: Tensor[(ROW*COL)*D1*...], or Iterable of same size tensors.
    :param original_nrow: original ROW
    :return: Tensor[(COL*ROW)*D1*...]
    """
    assert torch.is_tensor(tensors) or isinstance(tensors, Iterable)
    if not torch.is_tensor(tensors) and isinstance(tensors, Iterable):
        seen_size = None
        grid = []
        for tensor in tensors:
            if seen_size is None:
                seen_size = tensor.size()
                original_nrow = original_nrow or len(tensor)
            elif tensor.size() != seen_size:
                raise ValueError("expect all tensor in images have the same size.")
            grid.append(tensor)
        tensors = torch.cat(grid)

    assert original_nrow is not None

    cell_size = tensors.size()[1:]

    tensors = tensors.reshape(-1, original_nrow, *cell_size)
    tensors = torch.transpose(tensors, 0, 1)
    return torch.reshape(tensors, (-1, *cell_size))


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        remaining_steps = self.num_total_steps - self.step
        if self.step > 0:
            training_time_left = remaining_steps * (time_sofar / self.step)
        else:
            training_time_left = 0

   
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, 
                                sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        
       