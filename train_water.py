import argparse
import os
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np

import wasr.models as models
from wasr.train import LitModel
from wasr.utils import ModelExporter, load_weights
from datasets.WaterDataset import WaterDataset
from datasets.transforms import get_augmentation_transform, PytorchHubNormalization


# 默认配置
DEVICE_BATCH_SIZE = 4
NUM_CLASSES = 2
PATIENCE = 15
LOG_STEPS = 20
NUM_WORKERS = 4
NUM_GPUS = -1
RANDOM_SEED = 42
OUTPUT_DIR = 'output_water'
PRETRAINED_DEEPLAB = True
PRECISION = 32
MODEL = 'wasr_resnet101'
MONITOR_VAR = 'val/iou/water'
MONITOR_VAR_MODE = 'max'


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


class MetricsCallback(Callback):
    """自定义回调，记录训练和验证指标到CSV"""
    
    def __init__(self, save_path, model_info, args):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.args = args
        
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        self.rows = []
        self.epoch_start_time = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        """记录epoch开始时间"""
        self.epoch_start_time = time.time()
        
    def _get_metric(self, metrics, key_list):
        """辅助函数：从metrics字典中获取第一个存在的键值"""
        for key in key_list:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                return value
        return 0.0
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时 - 此时所有指标都已聚合完成"""
        metrics = trainer.callback_metrics
        
        # 获取训练指标（带_epoch后缀，因为是epoch级别聚合的）
        train_loss = self._get_metric(metrics, ['train/loss_epoch', 'train/loss'])
        train_precision = self._get_metric(metrics, ['train/precision'])
        train_recall = self._get_metric(metrics, ['train/recall'])
        train_f1 = self._get_metric(metrics, ['train/f1'])
        train_miou = self._get_metric(metrics, ['train/miou', 'train/iou/water_epoch', 'train/iou/water'])
        
        # 获取验证指标
        val_loss = self._get_metric(metrics, ['val/loss'])
        val_precision = self._get_metric(metrics, ['val/precision'])
        val_recall = self._get_metric(metrics, ['val/recall'])
        val_f1 = self._get_metric(metrics, ['val/f1'])
        val_miou = self._get_metric(metrics, ['val/miou', 'val/iou/water'])
        
        # 计算epoch时间和FPS
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        val_loader = trainer.val_dataloaders
        val_size = len(val_loader.dataset) if val_loader else 0
        fps = val_size / epoch_time if epoch_time > 0 else 0
        inference_time_ms = (epoch_time / len(val_loader)) * 1000 if val_loader and len(val_loader) > 0 else 0
        
        # 获取当前学习率
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0
        
        # 记录到CSV
        row = {
            'epoch': trainer.current_epoch + 1,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_precision:.6f}",
            'train_recall': f"{train_recall:.6f}",
            'train_f1': f"{train_f1:.6f}",
            'train_miou': f"{train_miou:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_precision': f"{val_precision:.6f}",
            'val_recall': f"{val_recall:.6f}",
            'val_f1': f"{val_f1:.6f}",
            'val_miou': f"{val_miou:.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        self.rows.append(row)
        
        # 打印当前epoch摘要
        print(f"\nEpoch {trainer.current_epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, mIoU: {train_miou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, mIoU: {val_miou:.4f}, FPS: {fps:.2f}")
        
    def on_fit_end(self, trainer, pl_module):
        """训练结束时保存CSV"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"\nTraining log saved to {self.save_path}")
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model Type: WaSR\n")
            f.write(f"Backbone: {self.args.model}\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Number of Classes: {self.args.num_classes}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Epochs: {self.args.epochs}\n")
            
            lr = getattr(self.args, 'lr', None) or getattr(self.args, 'learning_rate', 'N/A')
            f.write(f"Learning Rate: {lr}\n")
            
            optimizer = getattr(self.args, 'optimizer', 'N/A')
            f.write(f"Optimizer: {optimizer}\n")
            
            f.write(f"Pretrained: {self.args.pretrained}\n")


def get_arguments(input_args=None):
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(
        description="WaSR Water Segmentation Training (Binary Classification)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集配置
    parser.add_argument("--train_config", type=str, required=True,
                        help="Path to the training dataset YAML config.")
    parser.add_argument("--val_config", type=str, required=True,
                        help="Path to the validation dataset YAML config.")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=DEVICE_BATCH_SIZE,
                        help="Minibatch size per device.")
    parser.add_argument("--validation", action="store_true", default=True,
                        help="Enable validation and early stopping.")
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
                        help="Number of classes (2 for water/non-water).")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early stopping patience.")
    parser.add_argument("--log_steps", type=int, default=LOG_STEPS,
                        help="Logging interval.")
    parser.add_argument("--gpus", default=NUM_GPUS,
                        help="Number of GPUs or GPU IDs.")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="Data loading workers.")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility.")
    parser.add_argument("--pretrained", type=bool, default=PRETRAINED_DEEPLAB,
                        help="Use pretrained DeepLab weights.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for models and logs.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Experiment name.")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Path to pretrained weights (e.g., MaSTr1325 weights).")
    parser.add_argument("--model", type=str, choices=models.model_list, default=MODEL,
                        help="Model architecture.")
    parser.add_argument("--monitor_metric", type=str, default=MONITOR_VAR,
                        help="Metric to monitor for checkpointing.")
    parser.add_argument("--monitor_metric_mode", type=str, default=MONITOR_VAR_MODE, 
                        choices=['min', 'max'],
                        help="Optimization mode for monitored metric.")
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable data augmentation.")
    parser.add_argument("--precision", default=PRECISION, type=int, choices=[16, 32],
                        help="Floating point precision.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint.")
    
    # 水分割特有参数
    parser.add_argument("--water_class_weight", type=float, default=2.0,
                        help="Loss weight for water class (to handle class imbalance).")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone and only train segmentation head.")
    parser.add_argument("--adapt_weights", action="store_true", default=True,
                        help="Adapt pretrained 3-class weights to 2-class.")

    # LitModel 添加的参数（包括 --epochs, --lr 等）
    parser = LitModel.add_argparse_args(parser)
    args = parser.parse_args(input_args)

    return args


def adapt_pretrained_weights(state_dict, num_classes=2):
    """将预训练的3类权重适配到2类"""
    adapted_state_dict = {}
    
    for key, value in state_dict.items():
        if 'classifier' in key or 'fc' in key or 'conv' in key:
            if len(value.shape) > 0 and value.shape[0] == 3:
                print(f"Skipping {key}: {value.shape} -> adapting to {num_classes} classes")
                continue
        
        adapted_state_dict[key] = value
    
    return adapted_state_dict


def train_water(args):
    """水分割训练主函数"""
    
    # 设置随机种子
    args.random_seed = pl.seed_everything(args.random_seed)
    print(f"Random seed: {args.random_seed}")
    print(f"Training configuration: {args}")

    # 数据预处理
    normalize_t = PytorchHubNormalization()

    # 数据增强
    transform = None
    if not args.no_augmentation:
        transform = get_augmentation_transform()
        print("Data augmentation enabled")

    # 创建数据集
    print(f"\nLoading training data from: {args.train_config}")
    train_ds = WaterDataset(
        args.train_config, 
        transform=transform,
        normalize_t=normalize_t
    )
    print(f"Training samples: {len(train_ds)}")

    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers, 
        persistent_workers=args.workers > 0,
        drop_last=True,
        pin_memory=True
    )

    # 验证集
    val_dl = None
    if args.validation:
        print(f"\nLoading validation data from: {args.val_config}")
        val_ds = WaterDataset(
            args.val_config, 
            normalize_t=normalize_t, 
            include_original=True
        )
        print(f"Validation samples: {len(val_ds)}")
        
        val_dl = DataLoader(
            val_ds, 
            batch_size=args.batch_size, 
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=True
        )

    # 创建模型
    print(f"\nInitializing model: {args.model}")
    print(f"Number of classes: {args.num_classes}")
    
    model = models.get_model(
        args.model, 
        num_classes=args.num_classes, 
        pretrained=args.pretrained
    )

    # 获取模型信息
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

    # 加载预训练权重
    if args.pretrained_weights is not None:
        print(f"\nLoading pretrained weights from: {args.pretrained_weights}")
        state_dict = load_weights(args.pretrained_weights)
        
        if args.adapt_weights and args.num_classes == 2:
            print("Adapting weights from 3 classes to 2 classes...")
            state_dict = adapt_pretrained_weights(state_dict, num_classes=2)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
        
        print("Weights loaded successfully")

    # 冻结主干网络
    if args.freeze_backbone:
        print("\nFreezing backbone parameters...")
        for name, param in model.named_parameters():
            if 'backbone' in name or 'encoder' in name:
                param.requires_grad = False
                print(f"  Frozen: {name}")

    # 包装为Lightning模型
    model = LitModel(model, args.num_classes, args)

    # 日志记录
    logs_path = os.path.join(args.output_dir, 'logs')
    os.makedirs(logs_path, exist_ok=True)
    logger = pl_loggers.TensorBoardLogger(logs_path, args.model_name)
    logger.log_hyperparams(vars(args))

    # 创建CSV记录回调
    log_filename = f"wasr_{args.model}_training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_callback = MetricsCallback(
        os.path.join(args.output_dir, args.model_name, log_filename),
        model_info,
        args
    )
    metrics_callback.save_model_info()

    # 回调函数
    callbacks = [metrics_callback]
    
    if args.validation:
        # 早停
        if args.patience is not None:
            callbacks.append(EarlyStopping(
                monitor=args.monitor_metric, 
                patience=args.patience, 
                mode=args.monitor_metric_mode,
                verbose=True
            ))
        
        # 最佳模型保存
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints', args.model_name),
            save_last=True,
            save_top_k=1,
            monitor=args.monitor_metric,
            mode=args.monitor_metric_mode,
            filename='best-{epoch:02d}-{' + args.monitor_metric.replace('/', '_') + ':.4f}',
            verbose=True
        ))
        
        # 模型导出
        callbacks.append(ModelExporter())

    # 处理设备参数
    if args.gpus == -1:
        num_gpus = torch.cuda.device_count()
        devices = num_gpus if num_gpus > 0 else 1
    elif args.gpus == 0:
        devices = 1
    else:
        devices = int(args.gpus)
    
    # 判断使用 GPU 还是 CPU
    if args.gpus == 0 or (args.gpus == -1 and torch.cuda.device_count() == 0):
        accelerator = "cpu"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "gpu"
        strategy = "ddp" if devices > 1 else "auto"
    
    # 处理 resume_from_checkpoint
    trainer_kwargs = {
        "logger": logger,
        "accelerator": accelerator,
        "devices": devices,
        "max_epochs": args.epochs,
        "strategy": strategy,
        "callbacks": callbacks,
        "sync_batchnorm": True if devices > 1 else False,
        "log_every_n_steps": args.log_steps,
        "precision": args.precision,
        "gradient_clip_val": 1.0,
        "check_val_every_n_epoch": 1,
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    if args.resume_from is not None:
        trainer_kwargs["ckpt_path"] = args.resume_from
    
    # 创建训练器
    trainer = pl.Trainer(**trainer_kwargs)

    # 开始训练
    print("\n" + "="*50)
    print(f"Accelerator: {accelerator}, Devices: {devices}, Strategy: {strategy}")
    print("Starting training...")
    print("="*50 + "\n")
    
    trainer.fit(model, train_dl, val_dl)
    
    print("\nTraining completed!")
    print(f"Outputs saved to: {os.path.join(args.output_dir, args.model_name)}")


def main():
    args = get_arguments()
    train_water(args)


if __name__ == '__main__':
    main()