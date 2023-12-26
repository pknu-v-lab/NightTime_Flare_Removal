import torch
import torch.nn as nn


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
    

class PELoss(nn.Module):
    def __init__(
            self,
            alpha=0.85,
            no_ssim=False
    ):
        """
        photometric error loss
        """
        super().__init__()
        self.alpha = alpha
        self.no_ssim = no_ssim
        if self.no_ssim == False:
            self.ssim = SSIM()


    def forward(self, pred, target):
        abs_diff = torch.abs(target - pred)     # (B,3,H,W)
        l1_loss = abs_diff.mean(1, True)


        if self.no_ssim:
            pe_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            pe_loss = 0.85 * ssim_loss + 0.15 * l1_loss


        return torch.mean(pe_loss)