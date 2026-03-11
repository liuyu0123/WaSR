import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml


class WaterDataset(torch.utils.data.Dataset):
    """Water Segmentation Dataset for binary classification (water/non-water)
    
    Expected directory structure:
        dataset/
        ├── train/
        │   ├── images/
        │   │   ├── 0001.jpg
        │   │   ├── 0002.jpg
        │   │   └── ...
        │   └── masks/
        │       ├── 0001.png
        │       ├── 0002.png
        │       └── ...
        └── val/
            ├── images/
            └── masks/
    
    Expected YAML format:
        image_dir: path/to/images
        mask_dir: path/to/masks
    
    Mask format:
        - Grayscale image (0 = non-water/background, 1 = water)
        - Or binary image that will be thresholded (>127 = 1, else = 0)
    
    Args:
        dataset_file (str): Path to the dataset YAML configuration file.
        transform (optional): Transform to apply to image and masks (Albumentations).
        normalize_t (optional): Transform that normalizes the input image.
        include_original (optional): Include original (non-normalized) version of the image.
    """
    
    def __init__(self, dataset_file, transform=None, normalize_t=None, include_original=False):
        dataset_file = Path(dataset_file)
        self.dataset_dir = dataset_file.parent
        
        # Load YAML config
        with dataset_file.open('r') as file:
            data = yaml.safe_load(file)
        
        # Set data directories (支持相对路径和绝对路径)
        self.image_dir = Path(data['image_dir'])
        if not self.image_dir.is_absolute():
            self.image_dir = (self.dataset_dir / self.image_dir).resolve()
            
        self.mask_dir = Path(data['mask_dir'])
        if not self.mask_dir.is_absolute():
            self.mask_dir = (self.dataset_dir / self.mask_dir).resolve()
        
        # 自动扫描所有图像文件
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(valid_extensions)
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"[WaterDataset] Found {len(self.images)} images in {self.image_dir}")
        
        self.transform = transform
        self.normalize_t = normalize_t
        self.include_original = include_original

    def __len__(self):
        return len(self.images)

    def _find_mask_path(self, img_name):
        """根据图像名查找对应的mask路径"""
        base_name = os.path.splitext(img_name)[0]
        
        # 尝试常见mask扩展名
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        for ext in mask_extensions:
            mask_path = self.mask_dir / (base_name + ext)
            if mask_path.exists():
                return str(mask_path)
        
        # 如果没找到，返回默认的.png路径（用于报错提示）
        return str(self.mask_dir / (base_name + '.png'))

    def _read_mask(self, mask_path):
        """读取mask并转换为二分类格式 (0和1)"""
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # 二值化：>127为1（水），否则为0（非水）
        # 如果你的mask已经是0和1，这行不会改变值
        mask = (mask > 127).astype(np.float32)
        
        # 转换为 one-hot 格式: [H, W, 2]
        # 通道0: 非水背景, 通道1: 水
        mask = np.stack([1 - mask, mask], axis=-1).astype(np.float32)
        
        return mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = self.image_dir / img_name
        
        # 读取图像 (RGB)
        img = np.array(Image.open(img_path).convert('RGB'))
        img_original = img.copy()
        
        # 查找并读取mask
        mask_path = self._find_mask_path(img_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = self._read_mask(mask_path)
        
        # 数据打包
        data = {
            'image': img,
            'segmentation': mask  # 使用相同的key 'segmentation' 保持兼容性
        }

        # 应用数据增强 (Albumentations)
        if self.transform is not None:
            data = self.transform(data)
            img = data['image']

        # 应用归一化
        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # 默认: 除以255
            img = TF.to_tensor(img)

        # 构建输出
        features = {'image': img}
        labels = {}

        # 可选：包含原始图像（用于可视化）
        if self.include_original:
            features['image_original'] = torch.from_numpy(img_original.transpose(2, 0, 1))

        # 分割标签 [H, W, 2] -> [2, H, W]
        if 'segmentation' in data:
            labels['segmentation'] = torch.from_numpy(data['segmentation'].transpose(2, 0, 1))

        # 元数据
        metadata = {
            'img_name': os.path.splitext(img_name)[0],
            'image_filename': img_name,
            'mask_filename': os.path.basename(mask_path)
        }
        labels.update(metadata)

        return features, labels