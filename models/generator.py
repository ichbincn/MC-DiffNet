import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fic import FICModule


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, fic_module=None, timesteps=1000):
        super().__init__()

        if fic_module is None:
            raise ValueError("FIC 模块未传递！请在 train.py 调用 build_generator(fic_module=FIC实例)")

        self.fic_module = fic_module
        self.timesteps = timesteps
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.fic_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.decoder3 = self._conv_block(256, 128, upsample=False)
        self.decoder2 = self._conv_block(128, 64, upsample=False)
        self.decoder1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.time_embedding = nn.Linear(1, 256)

    def _conv_block(self, in_channels, out_channels, upsample=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if upsample:
            layers.insert(0, nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x, timesteps, rci):
        timesteps = timesteps.float()
        time_embed = self.time_embedding(timesteps.view(-1, 1)).view(-1, 256, 1, 1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc3 = enc3 + time_embed
        enc3 = self.fic_module(enc3, rci)
        enc3 = self.fic_conv(enc3)
        dec3 = self.decoder3(enc3)
        dec2 = self.decoder2(dec3 + enc2)
        dec1 = self.decoder1(dec2 + enc1)
        return torch.sigmoid(dec1)

class DiffusionModel(nn.Module):
    def __init__(self, unet, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.timesteps = timesteps

    def forward(self, x, t, rci):
        return self.unet(x, t, rci)

    def generate(self, x_noisy, timesteps, rci):
        for t in reversed(range(timesteps)):
            noise_pred = self.unet(x_noisy, torch.tensor([t], dtype=torch.float32, device=x_noisy.device), rci)
            x_noisy = (x_noisy - noise_pred) / torch.sqrt(1 - 0.1 * t)
        return x_noisy

def build_generator(fic_module, timesteps=1000):
    if fic_module is None:
        raise ValueError("FIC 模块未传递！请在 train.py 确保 fic_module 已正确初始化")
    unet = DiffusionUNet(in_channels=1, out_channels=1, fic_module=fic_module, timesteps=timesteps)
    return DiffusionModel(unet, timesteps=timesteps)