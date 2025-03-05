import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import numpy as np

to_tensor = transforms.ToTensor()
from data.transforms import get_transforms
from data.utils import load_image

def degrade_image(img, target_size=(256, 256)):
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        elif img.ndim != 3:
            raise ValueError(f"Unsupported tensor shape: {img.shape}, expected [C, H, W]")
        img = transforms.ToPILImage()(img)

    elif not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type: {type(img)}")

    img = img.resize(target_size, Image.LANCZOS)

    downsample = transforms.Resize((target_size[0] // 2, target_size[1] // 2))
    upsample = transforms.Resize(target_size)
    img = upsample(downsample(img))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    img_np = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, 0.02, img_np.shape)
    img_np = np.clip(img_np + noise, 0, 1)

    return Image.fromarray((img_np * 255).astype(np.uint8))

class UltrasoundDataset(Dataset):

    def __init__(self, root_dir, dataset_type="unpaired", mode="train"):
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.mode = mode
        self.data = []

        dataset_path = os.path.join(root_dir, dataset_type)

        if mode == "train":
            high_quality_dir = os.path.join(dataset_path, "high quality")
            low_quality_dir = os.path.join(dataset_path, "low quality")
            mask_dir = os.path.join(dataset_path, "mask")

            # **过滤掉 `.DS_Store` 和其他非图片文件**
            high_quality_files = sorted([f for f in os.listdir(high_quality_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if dataset_type == "unpaired":
                low_quality_files = sorted([f for f in os.listdir(low_quality_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.data = [(os.path.join(high_quality_dir, hq), os.path.join(low_quality_dir, lq), None)
                             for hq, lq in zip(high_quality_files, low_quality_files)]
            else:
                if not os.path.exists(low_quality_dir):
                    os.makedirs(low_quality_dir)

                low_quality_files = sorted([f for f in os.listdir(low_quality_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

                if len(low_quality_files) == 0:
                    print(f"⚠️  Low-quality directory `{low_quality_dir}` is empty, generating images...")
                    self._generate_low_quality_images(high_quality_dir, low_quality_dir)
                else:
                    print(f"✅ Low-quality directory `{low_quality_dir}` already has {len(low_quality_files)} files.")

                mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.data = [(os.path.join(high_quality_dir, hq), os.path.join(low_quality_dir, hq), os.path.join(mask_dir, mask))
                             for hq, mask in zip(high_quality_files, mask_files)]


        else:

            low_quality_dir = os.path.join(dataset_path, "LR")

            high_quality_dir = os.path.join(dataset_path, "HR")

            mask_dir = os.path.join(dataset_path, "mask")

            low_quality_files = sorted(
                [f for f in os.listdir(low_quality_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

            if dataset_type == "unpaired":

                self.data = [(os.path.join(low_quality_dir, lq), os.path.join(high_quality_dir, lq))

                             for lq in low_quality_files]

            else:

                if os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0:

                    mask_files = sorted(
                        [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

                    self.data = [(os.path.join(low_quality_dir, lq), os.path.join(high_quality_dir, lq),
                                  os.path.join(mask_dir, mask))

                                 for lq, mask in zip(low_quality_files, mask_files)]

                else:

                    print("⚠️ Warning: No masks found in test set, proceeding without mask.")

                    self.data = [(os.path.join(low_quality_dir, lq), os.path.join(high_quality_dir, lq), None)

                                 for lq in low_quality_files]

    def _generate_low_quality_images(self, high_quality_dir, low_quality_dir):
        """
        生成低质量图像，并存入 low_quality 目录
        """
        print("📢 Generating low-quality images for `with_mask` dataset...")

        for img_name in os.listdir(high_quality_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(high_quality_dir, img_name)
            try:
                img = Image.open(img_path).convert("L")
                degraded_img = degrade_image(img)
                save_path = os.path.join(low_quality_dir, img_name)
                degraded_img.save(save_path)
                print(f"✅ Saved degraded image: {save_path}")
            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")

        print("✅ Finished generating low-quality images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "train":
            hq_path, lq_path, mask_path = self.data[idx]

            hq_img = load_image(hq_path)
            if hq_img is None:
                print(f"⚠️ Skipping sample `{hq_path}` due to loading error.")
                return self.__getitem__((idx + 1) % len(self.data))  # 递归获取下一个样本

            hq_img = to_tensor(hq_img)

            if os.path.exists(lq_path):
                lq_img = load_image(lq_path)
                if lq_img is None:
                    print(f"⚠️ Warning: Failed to load `{lq_path}`, generating degraded image.")
                    lq_img = degrade_image(hq_img)
            else:
                print(f"⚠️ `{lq_path}` not found, generating degraded version.")
                lq_img = degrade_image(hq_img)

            lq_img = to_tensor(lq_img)

            mask_img = None
            if mask_path:
                mask_img = load_image(mask_path)
                if mask_img is None:
                    print(f"⚠️ Warning: Failed to load mask `{mask_path}`. Skipping mask.")
                else:
                    mask_img = to_tensor(mask_img)

            resize_transform = transforms.Resize((256, 256))
            lq_img = resize_transform(lq_img)
            hq_img = resize_transform(hq_img)
            if mask_img is not None:
                mask_img = resize_transform(mask_img)

            if self.dataset_type == "unpaired":
                return lq_img, hq_img
            else:
                return lq_img, hq_img, mask_img
        else:
            if self.dataset_type == "unpaired":
                lq_path, hq_path = self.data[idx]
                lq_img = to_tensor(load_image(lq_path))
                return lq_img
            else:
                lq_path, hq_path, mask_path = self.data[idx]
                lq_img = to_tensor(load_image(lq_path))
                mask_img = to_tensor(load_image(mask_path)) if mask_path else None
                return (lq_img, mask_img)

def get_dataloader(root_dir, dataset_type="unpaired", mode="train", batch_size=8, shuffle=True):
    dataset = UltrasoundDataset(root_dir, dataset_type, mode)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)