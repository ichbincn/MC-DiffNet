import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, feature_dim=128):
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ACPClassifier(nn.Module):
    def __init__(self, feature_dim=128, dataset_type="unpaired"):
        super(ACPClassifier, self).__init__()
        self.cnn = CNNFeatureExtractor(in_channels=1, feature_dim=feature_dim)
        self.dataset_type = dataset_type
        self.k = min(5 if dataset_type == "unpaired" else 7, feature_dim)

    def forward(self, x):
        features = self.cnn(x)
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).mean(dim=-1)
        features_np = features_flat.detach().cpu().numpy()
        k = min(self.k, B)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features_np)
        cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=x.device)
        rci = torch.zeros(B, C, device=x.device)
        for i in range(k):
            mask = (cluster_labels == i)
            if mask.sum() > 0:
                rci[mask] = features_flat[mask].mean(dim=0)
        assert rci.shape == (B, 128), f"错误：RCI 形状不匹配，期望 [{B}, 128]，得到 {rci.shape}"
        return rci.detach()
def build_classifier(feature_dim=128, dataset_type="unpaired"):
    return ACPClassifier(feature_dim=feature_dim, dataset_type=dataset_type)