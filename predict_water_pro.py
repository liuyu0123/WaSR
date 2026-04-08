import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import csv

from datasets.folder import FolderDataset
from datasets.transforms import PytorchHubNormalization
from wasr.inference import Predictor
import wasr.models as models
from wasr.utils import load_weights

WATER_CLASS = 1
BATCH_SIZE = 12
ARCHITECTURE = 'wasr_resnet50'


class SingleImageDataset(Dataset):
    """单张图片数据集"""
    def __init__(self, image_path, imu_path=None, normalize_t=None):
        self.image_path = Path(image_path).resolve()
        self.imu_path = Path(imu_path).resolve() if imu_path else None
        self.normalize_t = normalize_t
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        img = Image.open(self.image_path).convert('RGB')
        imu_mask = None
        if self.imu_path and self.imu_path.exists():
            imu_mask = Image.open(self.imu_path).convert('L')
        
        if self.normalize_t:
            img = self.normalize_t(img)
            if imu_mask is not None:
                imu_mask = self.normalize_t(imu_mask)
        
        metadata = {'image_path': [str(self.image_path)]}
        if imu_mask is not None:
            return (img, imu_mask), metadata
        return (img,), metadata


def get_arguments():
    parser = argparse.ArgumentParser(description="WaSR Network Inference Script")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input image or directory containing images.")
    parser.add_argument("--imu_dir", type=str, default=None, 
                        help="(optional) Path to IMU masks directory.")
    parser.add_argument("--gt_mask_dir", type=str, default=None,
                        help="(optional) Path to ground truth masks directory.")
    parser.add_argument("--architecture", type=str, choices=models.model_list, 
                        default=ARCHITECTURE, help="Model architecture.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights.")
    parser.add_argument("--output", type=str, default=None, 
                        help="Output directory for overlay images.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--strict_load", action='store_true', default=False)
    parser.add_argument("--skip_mismatch_layers", action='store_true', default=False)
    return parser.parse_args()


def load_weights_safe(model, weights_path, strict=False, skip_mismatch=False):
    """安全加载权重"""
    print(f"Loading weights from: {weights_path}")
    state_dict = load_weights(weights_path)
    model_state = model.state_dict()
    
    missing_keys = set(model_state.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
    size_mismatch = {}
    matched_keys = []
    
    for key in state_dict:
        if key in model_state:
            if state_dict[key].shape != model_state[key].shape:
                size_mismatch[key] = (state_dict[key].shape, model_state[key].shape)
            else:
                matched_keys.append(key)
    
    print(f"\n[Weight Analysis] Model: {len(model_state)} | Weights: {len(state_dict)} | Matched: {len(matched_keys)}")
    if missing_keys:
        print(f"  Missing: {len(missing_keys)} layers")
        if any('imu' in k.lower() for k in missing_keys):
            print("    -> Hint: IMU layers missing, use architecture without '_imu'")
    if size_mismatch:
        print(f"  Size mismatch: {len(size_mismatch)} layers")
        for k, (w, m) in list(size_mismatch.items())[:2]:
            print(f"    - {k}: weight {w} vs model {m}")
        if any('2049' in str(m) for w, m in size_mismatch.values()):
            print("    -> Critical: Use wasr_resnet101 (not _imu version)")
    
    if skip_mismatch and size_mismatch:
        for key in size_mismatch:
            del state_dict[key]
    
    try:
        model.load_state_dict(state_dict, strict=strict and not skip_mismatch)
        print("[Success] Weights loaded!")
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("[Partial Load] Some layers use random init")
    return model


def calculate_metrics(pred_mask, gt_mask):
    """计算分割指标"""
    if pred_mask.shape != gt_mask.shape:
        gt_mask = np.array(Image.fromarray(gt_mask).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST))
    
    pred_water = (pred_mask == WATER_CLASS).astype(np.uint8)
    gt_water = (gt_mask == WATER_CLASS).astype(np.uint8)
    
    tp = np.sum((pred_water == 1) & (gt_water == 1))
    fp = np.sum((pred_water == 1) & (gt_water == 0))
    fn = np.sum((pred_water == 0) & (gt_water == 1))
    tn = np.sum((pred_water == 0) & (gt_water == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {
        'image_name': '',  # 稍后填充
        'precision': precision, 
        'recall': recall, 
        'f1_score': f1,
        'miou': iou, 
        'accuracy': accuracy,
        '_tp': int(tp), '_fp': int(fp), '_fn': int(fn), '_tn': int(tn)  # 内部字段，不计入CSV
    }


def create_overlay_image(original_image, pred_mask, alpha=0.5):
    """创建叠加蒙版：原图 + 透明红色水区域"""
    orig_array = np.array(original_image)
    h, w = pred_mask.shape
    
    if orig_array.shape[:2] != (h, w):
        original_image = original_image.resize((w, h), Image.BILINEAR)
        orig_array = np.array(original_image)
    
    overlay = orig_array.copy()
    water_mask = (pred_mask == WATER_CLASS).astype(np.float32)
    
    if water_mask.sum() > 0:
        red_overlay = np.zeros_like(orig_array)
        red_overlay[:, :, 0] = 255
        alpha_mask = water_mask[:, :, np.newaxis] * alpha
        overlay = (orig_array * (1 - alpha_mask) + red_overlay * alpha_mask).astype(np.uint8)
    
    return Image.fromarray(overlay)


def find_gt_mask(gt_dir, img_stem):
    """查找真值mask，支持多种命名"""
    if not gt_dir or not gt_dir.exists():
        return None
    
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '']:
        candidate = gt_dir / f"{img_stem}{ext}"
        if candidate.exists():
            return candidate
    
    for gt_file in gt_dir.iterdir():
        if gt_file.stem == img_stem:
            return gt_file
    return None


def load_gt_mask(gt_path, pred_shape):
    """加载真值mask"""
    if not gt_path or not gt_path.exists():
        return None
    try:
        gt_img = Image.open(gt_path).convert('L')
        gt_array = np.array(gt_img)
        if gt_array.shape != pred_shape:
            gt_img = gt_img.resize((pred_shape[1], pred_shape[0]), Image.NEAREST)
            gt_array = np.array(gt_img)
        return (gt_array > 127).astype(np.uint8)
    except Exception as e:
        print(f"Error loading GT {gt_path}: {e}")
        return None


def export_predictions(preds, batch, args, all_metrics, batch_idx=0):
    """处理预测结果"""
    features, metadata = batch
    batch_metrics = []
    
    output_dir = Path(args.output) if args.output else None
    gt_dir = Path(args.gt_mask_dir) if args.gt_mask_dir else None
    input_path = Path(args.input).resolve()
    
    for i, pred_mask in enumerate(preds):
        img_path_str = metadata['image_path'][i]
        img_path = Path(img_path_str)
        
        if not img_path.is_absolute():
            img_path = input_path / img_path if input_path.is_dir() else input_path.parent / img_path
        
        img_name = img_path.stem
        
        if batch_idx == 0 and i == 0:
            print(f"\n[Debug] First image: {img_path}")
            print(f"  Exists: {img_path.exists()}")
            if gt_dir:
                test_gt = find_gt_mask(gt_dir, img_name)
                print(f"  GT found: {test_gt if test_gt else 'Not found'}")
        
        try:
            original_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\nError loading image: {e}")
            print(f"  Metadata path: {metadata['image_path'][i]}")
            print(f"  Resolved path: {img_path}")
            continue
        
        if gt_dir:
            gt_path = find_gt_mask(gt_dir, img_name)
            if gt_path:
                gt_mask = load_gt_mask(gt_path, pred_mask.shape)
                if gt_mask is not None:
                    metrics = calculate_metrics(pred_mask, gt_mask)
                    metrics['image_name'] = img_name
                    batch_metrics.append(metrics)
        
        if output_dir:
            try:
                overlay_img = create_overlay_image(original_img, pred_mask, alpha=0.5)
                if input_path.is_dir():
                    try:
                        rel_path = img_path.relative_to(input_path)
                    except ValueError:
                        rel_path = Path(img_path.name)
                    save_path = output_dir / rel_path.with_suffix('.png')
                else:
                    save_path = output_dir / f"{img_name}_overlay.png"
                
                save_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_img.save(str(save_path))
            except Exception as e:
                print(f"Warning: Failed to save overlay: {e}")
    
    return batch_metrics


def generate_report(all_metrics, output_path=None):
    """生成报告 - 只包含指定字段，过滤内部字段"""
    if not all_metrics:
        print("No metrics collected.")
        return
    
    fieldnames = ['image_name', 'precision', 'recall', 'f1_score', 'miou', 'accuracy']
    
    # 计算平均值
    avg_metrics = {
        'image_name': 'AVERAGE',
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'miou': np.mean([m['miou'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics])
    }
    
    # 过滤数据：只保留 fieldnames 中的字段，排除 _tp/_fp 等内部字段
    report_data = []
    for m in all_metrics:
        filtered = {k: v for k, v in m.items() if k in fieldnames}
        report_data.append(filtered)
    
    report_data.append(avg_metrics)
    
    if output_path:
        csv_path = Path(output_path) / 'evaluation_report.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_data)
        
        print(f"\nReport saved: {csv_path}")
        print(f"Images: {len(all_metrics)} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f} | F1: {avg_metrics['f1_score']:.4f} | mIoU: {avg_metrics['miou']:.4f}")
    else:
        print("\n" + "="*85)
        print(f"{'Image':<35} {'Precision':>9} {'Recall':>9} {'F1':>9} {'mIoU':>9} {'Acc':>9}")
        print("-"*85)
        for m in all_metrics:
            print(f"{m['image_name']:<35} {m['precision']:>9.4f} {m['recall']:>9.4f} {m['f1_score']:>9.4f} {m['miou']:>9.4f} {m['accuracy']:>9.4f}")
        print("-"*85)
        print(f"{'AVERAGE':<35} {avg_metrics['precision']:>9.4f} {avg_metrics['recall']:>9.4f} {avg_metrics['f1_score']:>9.4f} {avg_metrics['miou']:>9.4f} {avg_metrics['accuracy']:>9.4f}")
        print("="*85)


def predict(args):
    input_path = Path(args.input).resolve()
    
    if input_path.is_file():
        print(f"Processing single image: {input_path}")
        imu_path = Path(args.imu_dir) if args.imu_dir else None
        dataset = SingleImageDataset(input_path, imu_path, normalize_t=PytorchHubNormalization())
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        dataset = FolderDataset(str(input_path), args.imu_dir, normalize_t=PytorchHubNormalization())
    else:
        raise ValueError(f"Input not found: {input_path}")
    
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)
    
    print(f"Loading model: {args.architecture}...")
    model = models.get_model(args.architecture, num_classes=args.num_classes, pretrained=False)
    model = load_weights_safe(model, args.weights, strict=args.strict_load, skip_mismatch=args.skip_mismatch_layers)
    model.eval()
    predictor = Predictor(model, args.fp16)
    
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        print(f"Output: {args.output}")
    
    all_metrics = []
    print("Starting prediction...")
    for batch_idx, batch in enumerate(tqdm(iter(dl), total=len(dl))):
        features, _ = batch
        pred_masks = predictor.predict_batch(features)
        batch_metrics = export_predictions(pred_masks, batch, args, all_metrics, batch_idx)
        all_metrics.extend(batch_metrics)
    
    print(f"\nProcessed {len(all_metrics)} images successfully.")
    
    if args.gt_mask_dir:
        if all_metrics:
            generate_report(all_metrics, args.output)
        else:
            print(f"\nWarning: No metrics calculated.")
            print(f"Check GT masks in: {args.gt_mask_dir}")


def main():
    args = get_arguments()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()
    predict(args)
    print("\nDone!")


if __name__ == '__main__':
    main()