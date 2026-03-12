import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 假设这些模块在您的项目结构中存在
from datasets.folder import FolderDataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights

# Colors corresponding to each segmentation class
# 0: Obstacle (Yellow), 1: Water (Blue), 2: Sky (Purple)
# 对于二分类任务，只会用到前两种颜色
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],  # Class 0: Obstacle/Background
    [41, 167, 224],  # Class 1: Water
    [90, 75, 164]    # Class 2: Sky (用于多类任务)
], np.uint8)

# 水的叠加颜色：透明红色 (RGBA)
WATER_OVERLAY_COLOR = np.array([255, 0, 0, 128], np.uint8)  # 红色，50%透明度

BATCH_SIZE = 12
ARCHITECTURE = 'wasr_resnet101_imu'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network Inference Script")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing input images.")
    parser.add_argument("--imu_dir", type=str, default=None, help="(optional) Path to the directory containing input IMU masks.")
    parser.add_argument("--architecture", type=str, choices=models.model_list, default=ARCHITECTURE, help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for segmentation masks.")
    parser.add_argument("--overlay_output_dir", type=str, default=None, help="Output directory for overlay images (original + mask). If not set, will not generate overlays.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision for inference.")
    
    # ===== 关键修改：添加 num_classes 参数 =====
    parser.add_argument("--num_classes", type=int, default=2, help="Number of segmentation classes (default: 2 for water segmentation).")
    
    return parser.parse_args()

def create_overlay_image(original_image, pred_mask, water_class=1, alpha=0.5):
    """
    创建叠加蒙版图片：原图 + 透明红色水区域
    
    Args:
        original_image: PIL Image (RGB)
        pred_mask: numpy array, 预测类别索引
        water_class: 水类别的索引（默认为1）
        alpha: 透明度，0-1之间
    Returns:
        PIL Image (RGB)
    """
    # 转换为numpy数组
    orig_array = np.array(original_image)
    h, w = pred_mask.shape
    
    # 确保原图尺寸与mask一致（如果不一致则调整）
    if orig_array.shape[:2] != (h, w):
        original_image = original_image.resize((w, h), Image.BILINEAR)
        orig_array = np.array(original_image)
    
    # 创建输出图像（RGB）
    overlay = orig_array.copy()
    
    # 创建水区域的mask（二值）
    water_mask = (pred_mask == water_class).astype(np.uint8)
    
    # 如果存在水区域，则叠加红色
    if water_mask.sum() > 0:
        # 红色叠加：R通道增加，G和B通道减少
        # 方式1：半透明红色叠加
        red_overlay = np.zeros_like(orig_array)
        red_overlay[:, :, 0] = 255  # R通道全红
        red_overlay[:, :, 1] = 0    # G通道
        red_overlay[:, :, 2] = 0    # B通道
        
        # 使用alpha混合
        # overlay = orig * (1-alpha) + red * alpha
        alpha_mask = water_mask[:, :, np.newaxis] * alpha
        overlay = (orig_array * (1 - alpha_mask) + red_overlay * alpha_mask).astype(np.uint8)
    
    return Image.fromarray(overlay)

def export_predictions(preds, batch, output_dir, overlay_output_dir, num_classes, image_dir):
    features, metadata = batch
    
    for i, pred_mask in enumerate(preds):
        # ========== 1. 保存原始分割mask（黄蓝配色） ==========
        pred_mask_colored = SEGMENTATION_COLORS[pred_mask]
        mask_img = Image.fromarray(pred_mask_colored)
        
        out_path = output_dir / Path(metadata['image_path'][i]).with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(str(out_path))
        
        # ========== 2. 保存叠加蒙版图片（如果指定了overlay_output_dir） ==========
        if overlay_output_dir is not None:
            # 加载原始图片
            img_path = Path(image_dir) / metadata['image_path'][i]
            try:
                original_img = Image.open(img_path).convert('RGB')
                
                # 创建叠加图（水为透明红色，非水不绘制）
                overlay_img = create_overlay_image(original_img, pred_mask, water_class=1, alpha=0.5)
                
                # 保存叠加图
                overlay_path = overlay_output_dir / Path(metadata['image_path'][i]).with_suffix('.png')
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_img.save(str(overlay_path))
            except Exception as e:
                print(f"Warning: Failed to create overlay for {metadata['image_path'][i]}: {e}")

def predict(args):
    # 数据集
    dataset = FolderDataset(args.image_dir, args.imu_dir, normalize_t=PytorchHubNormalization())
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # 准备模型
    print(f"Loading model: {args.architecture} with {args.num_classes} classes...")
    
    # ===== 关键修改：传入 num_classes =====
    model = models.get_model(args.architecture, num_classes=args.num_classes, pretrained=False)
    
    # 加载权重
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    
    # 预测器
    predictor = Predictor(model, args.fp16)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果指定了overlay输出目录，也创建它
    overlay_output_dir = None
    if args.overlay_output_dir:
        overlay_output_dir = Path(args.overlay_output_dir)
        overlay_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Overlay images will be saved to: {overlay_output_dir}")

    print("Starting prediction...")
    for batch in tqdm(iter(dl), total=len(dl)):
        features, _ = batch
        pred_masks = predictor.predict_batch(features)
        export_predictions(pred_masks, batch, output_dir, overlay_output_dir, args.num_classes, args.image_dir)

def main():
    args = get_arguments()
    print(args)
    predict(args)

if __name__ == '__main__':
    main()