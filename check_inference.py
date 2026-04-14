import torch
from datasets.folder import FolderDataset
from datasets.transforms import PytorchHubNormalization

# 1. 加载数据
ds = FolderDataset(
    'D:/Files/GitProject/BiSeNet-ooooverflow-LY/dataset/water_seg2/test/images', 
    None, 
    normalize_t=PytorchHubNormalization()
)

features, _ = ds[0]
img = features['image']

print(f"图像 Shape: {img.shape}")
print(f"图像数据类型: {img.dtype}")
print(f"图像值范围: [{img.min():.4f}, {img.max():.4f}]")
print(f"图像均值: {img.mean():.4f}")

# 正常情况下：
# Shape: [3, H, W]
# dtype: torch.float32
# 范围: 大约在 [-2, 2] 之间 (因为经过了归一化)
# 均值: 应该接近 0
