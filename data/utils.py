import os
from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils


def load_image(img_path):
    if img_path is None or not os.path.exists(img_path):
        return None
    return Image.open(img_path).convert("L")


def save_image(tensor, path):
    vutils.save_image(tensor, path)


def check_dataset_structure(root_dir):
    required_dirs = [
        os.path.join(root_dir, "unpaired/high quality"),
        os.path.join(root_dir, "unpaired/low quality"),
        os.path.join(root_dir, "unpaired/mask"),
        os.path.join(root_dir, "with mask/high quality"),
        os.path.join(root_dir, "with mask/mask"),
        os.path.join(root_dir, "test/unpaired/LR"),
        os.path.join(root_dir, "test/unpaired/HR"),
        os.path.join(root_dir, "test/with mask/LR"),
        os.path.join(root_dir, "test/with mask/HR"),
    ]

    for d in required_dirs:
        if not os.path.exists(d):
            print(f"❌ 目录缺失: {d}")
            return False
    print("✅ 数据集结构完整")
    return True