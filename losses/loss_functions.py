import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from data.dataset import degrade_image
import torchvision.transforms as transforms
from data.dataset import degrade_image
import torch.nn.functional as F
import PIL

print(f"Using F.l1_loss function: {F.l1_loss}")
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        mu1 = TF.gaussian_blur(img1, self.window_size)
        mu2 = TF.gaussian_blur(img2, self.window_size)
        sigma1_sq = TF.gaussian_blur(img1 ** 2, self.window_size) - mu1 ** 2
        sigma2_sq = TF.gaussian_blur(img2 ** 2, self.window_size) - mu2 ** 2
        sigma12 = TF.gaussian_blur(img1 * img2, self.window_size) - mu1 * mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class LGenLoss(nn.Module):
    def __init__(self):
        super(LGenLoss, self).__init__()

    def forward(self, x_high, x_high_hat):
        return F.mse_loss(x_high, x_high_hat)

class LDegradeLoss(nn.Module):
    def __init__(self):
        super(LDegradeLoss, self).__init__()
        self.to_tensor = transforms.ToTensor()

    def forward(self, x_low, x_high):
        simulated_low = degrade_image(x_high)
        if isinstance(simulated_low, torch.Tensor):
            simulated_low = simulated_low.to(x_low.device)
        elif isinstance(simulated_low, np.ndarray):
            simulated_low = torch.tensor(simulated_low, dtype=torch.float32).permute(2, 0, 1)
            simulated_low = simulated_low.unsqueeze(0).to(x_low.device)
        else:
            simulated_low = self.to_tensor(simulated_low).unsqueeze(0).to(x_low.device)

        simulated_low = simulated_low.expand_as(x_low)
        loss = F.mse_loss(x_low, simulated_low)
        return loss

class LCycleLoss(nn.Module):
    def __init__(self):
        super(LCycleLoss, self).__init__()

    def forward(self, x_low, x_high_hat_low):

        if isinstance(x_high_hat_low, PIL.Image.Image):
            x_high_hat_low = transforms.ToTensor()(x_high_hat_low)
            x_high_hat_low = x_high_hat_low.unsqueeze(0).to(x_low.device)

        if x_high_hat_low.ndim == 3:
            x_high_hat_low = x_high_hat_low.unsqueeze(0)

        return F.l1_loss(x_low, x_high_hat_low)

class MaskSSIMLoss(nn.Module):
    def __init__(self):
        super(MaskSSIMLoss, self).__init__()
        self.ssim = SSIMLoss()

    def forward(self, x_high_masked, x_high_hat_masked):
        return self.ssim(x_high_masked, x_high_hat_masked)

def build_losses():
    return {
        "gen": LGenLoss(),
        "degrade": LDegradeLoss(),
        "cycle": LCycleLoss(),
        "mask_ssim": MaskSSIMLoss()
    }