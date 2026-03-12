import torch
from datasets.WaterDataset import WaterDataset
import wasr.models as models

# 1. 加载数据
ds = WaterDataset('configs/waterseg_train.yaml')
features, labels = ds[0]

print("=== 数据检查 ===")
print(f"图像形状: {features['image'].shape}")
print(f"标签形状: {labels['segmentation'].shape}")
print(f"标签唯一值: {torch.unique(labels['segmentation'])}")
print(f"水像素比例: {(labels['segmentation'] == 1).float().mean():.2%}")

# 2. 测试模型前向传播
model = models.get_model('wasr_resnet50', num_classes=2, pretrained=True)
model.eval()

# 添加 batch 维度
batch_features = {'image': features['image'].unsqueeze(0)}
with torch.no_grad():
    output = model(batch_features)
    logits = output['out']
    
print("\n=== 模型输出检查 ===")
print(f"Logits 形状: {logits.shape}")
print(f"Logits 值范围: [{logits.min():.4f}, {logits.max():.4f}]")

# 3. 检查预测结果
preds = logits.argmax(1)
print(f"预测形状: {preds.shape}")
print(f"预测唯一值: {torch.unique(preds)}")
print(f"预测为水的像素: {(preds == 1).sum().item()} / {preds.numel()}")
