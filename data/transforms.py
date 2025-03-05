import random
import torchvision.transforms as transforms
from PIL import Image, ImageFilter, ImageOps


def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def degrade_image(img, noise_intensity=(5, 15)):
    downsample = transforms.Resize((img.size[1] // 2, img.size[0] // 2))
    upsample = transforms.Resize((img.size[1], img.size[0]))
    img = upsample(downsample(img)).convert("L")
    noise = Image.effect_noise(img.size, random.uniform(*noise_intensity))
    img = Image.blend(img, noise, alpha=0.3)
    return img


def binarize_mask(mask, threshold=128):
    mask = mask.convert("L")
    mask = mask.point(lambda p: 255 if p > threshold else 0)
    return mask