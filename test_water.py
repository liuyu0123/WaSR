import argparse
import os
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

import wasr.models as models
from wasr.utils import load_weights
from datasets.WaterDataset import WaterDataset
from datasets.transforms import PytorchHubNormalization


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def compute_confusion_matrix(preds, labels, num_classes):
    """计算混淆矩阵"""
    device = preds.device
    
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    
    valid_mask = (labels_flat >= 0) & (labels_flat < num_classes)
    preds_valid = preds_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]
    
    cm = torch.zeros(num_classes, num_classes, device=device)
    
    for t in range(num_classes):
        for p in range(num_classes):
            cm[t, p] += ((labels_valid == t) & (preds_valid == p)).sum()
    
    return cm


def compute_metrics_from_cm(cm, num_classes):
    """基于混淆矩阵计算 Precision, Recall, F1, IoU"""
    eps = 1e-7
    
    precisions = []
    recalls = []
    f1s = []
    ious = []
    
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        
        precisions.append(precision.item())
        recalls.append(recall.item())
        f1s.append(f1.item())
        ious.append(iou.item())
    
    return {
        'obstacle_precision': precisions[0],
        'obstacle_recall': recalls[0],
        'obstacle_f1': f1s[0],
        'obstacle_iou': ious[0],
        'water_precision': precisions[1] if num_classes > 1 else 0,
        'water_recall': recalls[1] if num_classes > 1 else 0,
        'water_f1': f1s[1] if num_classes > 1 else 0,
        'water_iou': ious[1] if num_classes > 1 else 0,
        'miou': sum(ious) / len(ious),
        'mean_precision': sum(precisions) / len(precisions),
        'mean_recall': sum(recalls) / len(recalls),
        'mean_f1': sum(f1s) / len(f1s),
    }


def compute_pixel_accuracy(preds, labels):
    """计算像素准确率"""
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total


def test_model(model, test_loader, device, num_classes=2):
    """测试模型并计算指标"""
    model.eval()
    
    total_cm = torch.zeros(num_classes, num_classes, device=device)
    total_pixels = 0
    correct_pixels = 0
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 处理 batch 格式 - WaterDataset 返回 (features_dict, labels_dict)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                features, labels = batch
            else:
                features = batch
                labels = batch
            
            # 从 features 字典中获取图像张量
            if isinstance(features, dict):
                if 'image' in features:
                    images = features['image']
                elif 'input' in features:
                    images = features['input']
                elif 'features' in features:
                    images = features['features']
                else:
                    # 如果找不到，尝试获取第一个张量类型的值
                    for key, value in features.items():
                        if isinstance(value, torch.Tensor):
                            images = value
                            print(f"[DEBUG] Using features['{key}'] as input, shape: {value.shape}")
                            break
                    else:
                        raise ValueError(f"Cannot find image tensor in features. Keys: {list(features.keys())}")
            else:
                images = features
                print(f"[DEBUG] features is not dict, type: {type(features)}, shape: {features.shape if hasattr(features, 'shape') else 'N/A'}")
            
            # 从 labels 字典中获取分割标签
            if isinstance(labels, dict):
                if 'segmentation' in labels:
                    labels_seg = labels['segmentation']
                elif 'mask' in labels:
                    labels_seg = labels['mask']
                elif 'label' in labels:
                    labels_seg = labels['label']
                else:
                    for key, value in labels.items():
                        if isinstance(value, torch.Tensor):
                            labels_seg = value
                            break
                    else:
                        raise ValueError(f"Cannot find segmentation tensor in labels. Keys: {list(labels.keys())}")
            else:
                labels_seg = labels
            
            # 移动数据到设备
            images = images.to(device)
            labels_seg = labels_seg.to(device)
            
            start_time = time.time()
            
            # 前向传播 - 模型期望字典格式输入
            # 根据错误信息，模型期望 {'image': tensor} 格式
            if isinstance(images, torch.Tensor):
                # 如果 images 是张量，包装成字典
                input_dict = {'image': images}
            else:
                input_dict = images
            
            # 调试信息
            if batch_idx == 0:
                print(f"[DEBUG] Input type: {type(input_dict)}")
                if isinstance(input_dict, dict):
                    for k, v in input_dict.items():
                        print(f"[DEBUG]   {k}: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
            
            out = model(input_dict)
            logits = out['out']
            
            # 获取预测
            if labels_seg.dim() == 3:
                # 索引格式 [B, H, W]
                labels_size = (labels_seg.size(1), labels_seg.size(2))
                labels_hard = labels_seg
            else:
                # One-hot 格式 [B, C, H, W]
                labels_size = (labels_seg.size(2), labels_seg.size(3))
                labels_hard = labels_seg.argmax(1)
            
            # 调整 logits 大小以匹配标签
            logits = F.interpolate(logits, size=labels_size, mode='bilinear', align_corners=False)
            preds = logits.argmax(1)
            
            batch_time = time.time() - start_time
            inference_times.append(batch_time)
            
            batch_cm = compute_confusion_matrix(preds, labels_hard, num_classes)
            total_cm += batch_cm
            
            correct_pixels += (preds == labels_hard).sum().item()
            total_pixels += labels_hard.numel()
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    metrics = compute_metrics_from_cm(total_cm, num_classes)
    metrics['pixel_accuracy'] = pixel_acc
    
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    metrics['avg_inference_time_ms'] = avg_inference_time * 1000
    
    return metrics


def save_test_results(metrics, model_info, args, output_path):
    """保存测试结果到CSV"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    header = [
        'model', 'num_classes', 'pixel_accuracy', 
        'obstacle_precision', 'obstacle_recall', 'obstacle_f1', 'obstacle_iou',
        'water_precision', 'water_recall', 'water_f1', 'water_iou',
        'mean_precision', 'mean_recall', 'mean_f1', 'miou',
        'avg_inference_time_ms', 'test_samples', 'total_params', 'model_size_mb'
    ]
    
    test_images_path = Path(args.test_images)
    if test_images_path.is_dir():
        num_test_samples = len(list(test_images_path.glob('*.jpg')) + 
                              list(test_images_path.glob('*.png')) + 
                              list(test_images_path.glob('*.jpeg')))
    else:
        num_test_samples = 0
    
    row = {
        'model': args.model,
        'num_classes': args.num_classes,
        'pixel_accuracy': f"{metrics['pixel_accuracy']:.6f}",
        'obstacle_precision': f"{metrics['obstacle_precision']:.6f}",
        'obstacle_recall': f"{metrics['obstacle_recall']:.6f}",
        'obstacle_f1': f"{metrics['obstacle_f1']:.6f}",
        'obstacle_iou': f"{metrics['obstacle_iou']:.6f}",
        'water_precision': f"{metrics['water_precision']:.6f}",
        'water_recall': f"{metrics['water_recall']:.6f}",
        'water_f1': f"{metrics['water_f1']:.6f}",
        'water_iou': f"{metrics['water_iou']:.6f}",
        'mean_precision': f"{metrics['mean_precision']:.6f}",
        'mean_recall': f"{metrics['mean_recall']:.6f}",
        'mean_f1': f"{metrics['mean_f1']:.6f}",
        'miou': f"{metrics['miou']:.6f}",
        'avg_inference_time_ms': f"{metrics['avg_inference_time_ms']:.4f}",
        'test_samples': num_test_samples,
        'total_params': model_info['total_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
    }
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"\nTest results saved to: {output_path}")
    
    # 保存详细报告
    report_path = output_path.parent / f"{output_path.stem}_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Water Segmentation Model Test Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Test Images: {args.test_images}\n")
        f.write(f"Test Masks: {args.test_masks}\n")
        f.write(f"Num Classes: {args.num_classes}\n\n")
        
        f.write("Model Info:\n")
        f.write(f"  Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"  Model Size: {model_info['model_size_mb']:.2f} MB\n\n")
        
        f.write("Test Results:\n")
        f.write(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
        f.write(f"  mIoU: {metrics['miou']:.4f}\n\n")
        
        f.write("Obstacle Class (Class 0):\n")
        f.write(f"  Precision: {metrics['obstacle_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['obstacle_recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['obstacle_f1']:.4f}\n")
        f.write(f"  IoU: {metrics['obstacle_iou']:.4f}\n\n")
        
        f.write("Water Class (Class 1):\n")
        f.write(f"  Precision: {metrics['water_precision']:.4f}\n")
        f.write(f"  Recall: {metrics['water_recall']:.4f}\n")
        f.write(f"  F1 Score: {metrics['water_f1']:.4f}\n")
        f.write(f"  IoU: {metrics['water_iou']:.4f}\n\n")
        
        f.write("Mean Metrics:\n")
        f.write(f"  Mean Precision: {metrics['mean_precision']:.4f}\n")
        f.write(f"  Mean Recall: {metrics['mean_recall']:.4f}\n")
        f.write(f"  Mean F1: {metrics['mean_f1']:.4f}\n\n")
        
        f.write(f"Average Inference Time: {metrics['avg_inference_time_ms']:.2f} ms/batch\n")
        f.write(f"Test Samples: {num_test_samples}\n")
        f.write("=" * 60 + "\n")
    
    print(f"Detailed report saved to: {report_path}")


def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(
        description="WaSR Water Segmentation Model Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型配置
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--model", type=str, choices=models.model_list, default='wasr_resnet101',
                        help="Model architecture.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes (2 for water/non-water).")
    
    # 测试数据
    parser.add_argument("--test-images", type=str, required=True,
                        help="Path to test images directory.")
    parser.add_argument("--test-masks", type=str, required=True,
                        help="Path to test masks directory.")
    
    # 测试参数
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for testing.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Data loading workers.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Full path for output CSV file (including filename). If not specified, auto-generated.")
    parser.add_argument("--output-dir", type=str, default='output_water/test_results',
                        help="Output directory for test results (used when --output-path is not specified).")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output file name (default: auto-generated based on model name).")
    
    # 设备
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU for inference if available.")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32],
                        help="Floating point precision.")

    return parser.parse_args()


def main():
    args = get_arguments()
    
    # 设置设备
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print(f"\nTest configuration: {args}")
    
    # 创建模型
    print(f"\nInitializing model: {args.model}")
    print(f"Number of classes: {args.num_classes}")
    
    model = models.get_model(
        args.model, 
        num_classes=args.num_classes, 
        pretrained=False
    )
    
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 加载权重
    print(f"\nLoading model weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    state_dict = load_weights(args.model_path)
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
        print("Extracted 'model' state_dict from checkpoint")
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    print("Model weights loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    # 创建测试数据集
    print(f"\nLoading test data:")
    print(f"  Images: {args.test_images}")
    print(f"  Masks:  {args.test_masks}")
    
    # 创建临时YAML配置文件
    import tempfile
    import yaml
    test_config = {
        'image_dir': args.test_images,
        'mask_dir': args.test_masks,
        'num_classes': args.num_classes
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        test_config_path = f.name
    
    try:
        normalize_t = PytorchHubNormalization()
        
        test_ds = WaterDataset(
            test_config_path,
            normalize_t=normalize_t,
            include_original=False
        )
        print(f"Test samples: {len(test_ds)}")
        
        test_dl = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # 运行测试
        print("\n" + "=" * 50)
        print("Starting evaluation...")
        print("=" * 50)
        
        start_time = time.time()
        metrics = test_model(model, test_dl, device, args.num_classes)
        total_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("Test Results:")
        print("=" * 50)
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"mIoU: {metrics['miou']:.4f}")
        print(f"\nWater Class:")
        print(f"  Precision: {metrics['water_precision']:.4f}")
        print(f"  Recall:    {metrics['water_recall']:.4f}")
        print(f"  F1 Score:  {metrics['water_f1']:.4f}")
        print(f"  IoU:       {metrics['water_iou']:.4f}")
        print(f"\nObstacle Class:")
        print(f"  Precision: {metrics['obstacle_precision']:.4f}")
        print(f"  Recall:    {metrics['obstacle_recall']:.4f}")
        print(f"  F1 Score:  {metrics['obstacle_f1']:.4f}")
        print(f"  IoU:       {metrics['obstacle_iou']:.4f}")
        print(f"\nMean Metrics:")
        print(f"  Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall:    {metrics['mean_recall']:.4f}")
        print(f"  Mean F1:        {metrics['mean_f1']:.4f}")
        print(f"\nAverage Inference Time: {metrics['avg_inference_time_ms']:.2f} ms/batch")
        print(f"Total Evaluation Time: {total_time:.2f} seconds")
        print("=" * 50)
        
        # 确定输出路径
        if args.output_path:
            output_path = args.output_path
        else:
            if args.output_name is None:
                model_name = Path(args.model_path).stem
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                output_name = f"test_{model_name}_{timestamp}.csv"
            else:
                output_name = args.output_name
            
            output_path = os.path.join(args.output_dir, output_name)
        
        save_test_results(metrics, model_info, args, output_path)
        
    finally:
        if os.path.exists(test_config_path):
            os.remove(test_config_path)
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main()