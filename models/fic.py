import torch
import torch.nn as nn


class FICModule(nn.Module):
    def __init__(self, feature_dim=128, target_dim=256):
        super(FICModule, self).__init__()
        self.rci_proj = nn.Linear(feature_dim, target_dim)  # `[B, 128] -> [B, 256]`
        self.mlp_fusion = nn.Sequential(
            nn.Linear(target_dim + target_dim, target_dim),  # `[B, 512] -> [B, 256]`
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
        )

    def forward(self, ef_lf, rci):
        B, C, H, W = ef_lf.shape
        if rci.ndim != 2 or rci.shape[1] != 128:
            raise ValueError(f"错误：RCI 形状应为 [B, 128]，但得到 {rci.shape}")
        rci = rci.contiguous().clone().detach().to(torch.float32).to(ef_lf.device)
        rci = self.rci_proj(rci)
        rci = rci.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        fusion = torch.cat([ef_lf, rci], dim=1)
        fusion = fusion.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        fusion = self.mlp_fusion(fusion)
        fusion = fusion.view(B, H, W, -1).permute(0, 3, 1, 2)
        return fusion

def build_fic(feature_dim=256):
    return FICModule(feature_dim=feature_dim)