import argparse
import os
import csv
import time
import tempfile
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


class IntervalCheckpoint(Callback):
    """分步保存回调：每N个epoch保存一次中间模型"""
    
    def __init__(self, save_dir, filename_prefix, interval):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename_prefix = filename_prefix
        self.interval = interval
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, pl_module):
        if self.interval > 0 and (trainer.current_epoch + 1) % self.interval == 0:
            epoch = trainer.current_epoch + 1
            filename = f"{self.filename_prefix}_epoch{epoch}.ckpt"
            save_path = self.save_dir / filename
            
            trainer.save_checkpoint(str(save_path))
            print(f"  [Interval Save] Epoch {epoch} model saved to {save_path}")


class MetricsCallback(Callback):
    """自定义回调，实时记录训练和验证指标到CSV（每个epoch追加写入）"""
    
    def __init__(self, save_path, model_info, args, debug=True):
        super().__init__()
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.args = args
        self.debug = debug  # 调试模式，打印可用指标
        
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_loss', 'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        
        self.epoch_start_time = None
        self.epoch_val_time = 0
        self.first_epoch = True
        
        # 立即创建文件并写入表头（覆盖模式）
        with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
        print(f"Log file initialized: {self.save_path}")
        
    def _get_metric(self, metrics, key_list):
        """辅助函数：从metrics字典中获取第一个存在的键值"""
        for key in key_list:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                # 确保值是数字
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.0
        
    def on_train_epoch_start(self, trainer, pl_module):
        """记录epoch开始时间"""
        self.epoch_start_time = time.time()
        self.epoch_val_time = 0
        
    def on_validation_epoch_start(self, trainer, pl_module):
        """记录验证开始时间（用于计算推理时间）"""
        self.val_start_time = time.time()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时 - 此时所有指标都已聚合完成，立即写入CSV"""
        metrics = trainer.callback_metrics
        
        # 调试模式：打印所有可用的指标键（仅第一个epoch）
        if self.debug and self.first_epoch:
            print(f"\n[DEBUG] Available callback_metrics keys: {sorted(list(metrics.keys()))}")
            # 也检查 logged_metrics
            if hasattr(trainer, 'logged_metrics'):
                print(f"[DEBUG] Available logged_metrics keys: {sorted(list(trainer.logged_metrics.keys()))}")
            self.first_epoch = False
        
        # 计算验证耗时
        if hasattr(self, 'val_start_time'):
            self.epoch_val_time = time.time() - self.val_start_time
        
        # 获取训练指标 - 尝试多种可能的键名（包括带_epoch后缀和不带的）
        train_loss = self._get_metric(metrics, [
            'train/loss_epoch', 'train/loss', 'loss_epoch', 'loss'
        ])
        train_precision = self._get_metric(metrics, [
            'train/precision_epoch', 'train/precision', 'precision_epoch', 'precision'
        ])
        train_recall = self._get_metric(metrics, [
            'train/recall_epoch', 'train/recall', 'recall_epoch', 'recall'
        ])
        train_f1 = self._get_metric(metrics, [
            'train/f1_epoch', 'train/f1', 'f1_epoch', 'f1'
        ])
        train_miou = self._get_metric(metrics, [
            'train/miou_epoch', 'train/miou', 'train/iou/water_epoch', 'train/iou/water', 
            'miou_epoch', 'miou', 'iou/water_epoch', 'iou/water'
        ])
        
        # 获取验证指标
        val_loss = self._get_metric(metrics, [
            'val/loss_epoch', 'val/loss', 'validation/loss_epoch', 'validation/loss'
        ])
        val_precision = self._get_metric(metrics, [
            'val/precision_epoch', 'val/precision', 'validation/precision_epoch', 'validation/precision'
        ])
        val_recall = self._get_metric(metrics, [
            'val/recall_epoch', 'val/recall', 'validation/recall_epoch', 'validation/recall'
        ])
        val_f1 = self._get_metric(metrics, [
            'val/f1_epoch', 'val/f1', 'validation/f1_epoch', 'validation/f1'
        ])
        val_miou = self._get_metric(metrics, [
            'val/miou_epoch', 'val/miou', 'val/iou/water_epoch', 'val/iou/water',
            'validation/miou_epoch', 'validation/miou', 'validation/iou/water_epoch', 'validation/iou/water'
        ])
        
        # 计算时间指标
        val_loader = trainer.val_dataloaders
        val_size = len(val_loader.dataset) if val_loader else 0
        
        # FPS基于验证集推理时间计算
        fps = val_size / self.epoch_val_time if self.epoch_val_time > 0 else 0
        inference_time_ms = (self.epoch_val_time / len(val_loader)) * 1000 if val_loader and len(val_loader) > 0 else 0
        
        # 获取当前学习率
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0
        
        # 构建行数据
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
        
        # 立即追加写入CSV（实时写入，不缓存）
        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)
        
        # 打印当前epoch摘要
        print(f"\nEpoch {trainer.current_epoch + 1} Summary:")
        print(f"  Train -> Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, mIoU: {train_miou:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, mIoU: {val_miou:.4f}, FPS: {fps:.2f}")
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
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
            f.write(f"Model Name Prefix: {self.args.model_name}\n")
            f.write(f"Save Interval: {self.args.save_interval}\n")


def get_arguments(input_args=None):
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(
        description="WaSR Water Segmentation Training (支持直接路径或YAML配置)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集配置 - 方式1：YAML文件（向后兼容）
    parser.add_argument("--train_config", type=str, default=None,
                        help="训练数据集YAML配置文件路径（与--images/--masks互斥）")
    parser.add_argument("--val_config", type=str, default=None,
                        help="验证数据集YAML配置文件路径（与--val-images/--val-masks互斥）")
    
    # 数据集配置 - 方式2：直接路径（新增）
    parser.add_argument("--images", type=str, default=None,
                        help="训练图像目录路径")
    parser.add_argument("--masks", type=str, default=None,
                        help="训练掩码目录路径")
    parser.add_argument("--val-images", type=str, dest='val_images', default=None,
                        help="验证图像目录路径")
    parser.add_argument("--val-masks", type=str, dest='val_masks', default=None,
                        help="验证掩码目录路径")
    
    # 新增：保存路径和文件名控制参数
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型保存目录（默认: output_dir/checkpoints/model_name）")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="日志保存目录（默认: output_dir/logs）")
    parser.add_argument("--model-name", type=str, default=None,
                        help="模型保存文件名前缀（默认: 使用--model_name参数值）")
    parser.add_argument("--log-name", type=str, default=None,
                        help="日志文件名前缀（默认: wasr_{model}_training_log_时间戳）")
    parser.add_argument("--save-interval", type=int, default=0,
                        help="分步保存频率，每N个epoch保存一次中间模型，0为不保存（默认: 0）")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=DEVICE_BATCH_SIZE,
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
                        help="Output directory for models and logs (当未指定--model-dir/--log-dir时使用).")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Experiment name (用于TensorBoard等，若未指定--model-name则也用于文件名).")
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
    
    # 参数验证：必须提供一种数据配置方式
    has_yaml = args.train_config is not None or args.val_config is not None
    has_direct = args.images is not None or args.masks is not None
    
    if not has_yaml and not has_direct:
        parser.error("必须提供数据集配置：要么使用 --train_config/--val_config (YAML方式)，要么使用 --images/--masks/--val-images/--val-masks (直接路径方式)")
    
    if has_direct and (args.images is None or args.masks is None):
        parser.error("使用直接路径方式时，--images 和 --masks 必须同时指定")
    
    if has_direct and args.validation and (args.val_images is None or args.val_masks is None):
        parser.error("使用直接路径方式且启用验证时，--val-images 和 --val-masks 必须同时指定")
    
    # 设置默认名称
    if args.model_name is None:
        args.model_name = "wasr_experiment"
    
    # 如果未指定 --model-name，沿用 --model_name（实验名）
    if args.model_name is None:
        args.model_name = args.model_name
    
    # 如果未指定 --log-name，生成默认带时间戳的名称
    if args.log_name is None:
        args.log_name = f"wasr_{args.model}_training_log_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 清理 .csv 后缀（如果有），我们会在代码中自动添加
    if args.log_name.endswith('.csv'):
        args.log_name = args.log_name[:-4]
    
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


def create_temp_yaml_config(image_dir, mask_dir):
    """
    为直接路径模式创建临时 YAML 配置文件
    WaterDataset 期望 YAML 文件路径，因此需要动态创建
    """
    config_content = f"image_dir: {image_dir}\nmask_dir: {mask_dir}\n"
    
    # 创建临时文件，使用 .yaml 后缀
    fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='wasr_config_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(config_content)
        return temp_path
    except Exception as e:
        os.close(fd)
        raise e


def train_water(args):
    """水分割训练主函数"""
    
    # 设置随机种子
    pl.seed_everything(args.random_seed)
    print(f"Random seed: {args.random_seed}")
    print(f"Configuration: {args}")

    # 数据预处理
    normalize_t = PytorchHubNormalization()

    # 数据增强
    transform = None
    if not args.no_augmentation:
        transform = get_augmentation_transform()
        print("Data augmentation enabled")

    # 确定数据集配置来源并创建临时配置文件（如果需要）
    temp_train_config = None
    temp_val_config = None
    
    if args.images is not None:  # 使用直接路径模式
        print(f"\n使用直接路径模式:")
        print(f"  训练图像: {args.images}")
        print(f"  训练掩码: {args.masks}")
        
        # 创建临时 YAML 配置文件
        temp_train_config = create_temp_yaml_config(args.images, args.masks)
        train_config = temp_train_config
        
        if args.validation and args.val_images:
            print(f"  验证图像: {args.val_images}")
            print(f"  验证掩码: {args.val_masks}")
            temp_val_config = create_temp_yaml_config(args.val_images, args.val_masks)
            val_config = temp_val_config
        else:
            val_config = None
    else:  # 使用 YAML 配置模式
        print(f"\n使用 YAML 配置模式:")
        print(f"  训练配置: {args.train_config}")
        train_config = args.train_config
        val_config = args.val_config if args.validation else None

    try:
        # 创建数据集
        print(f"\nLoading training data...")
        train_ds = WaterDataset(
            train_config, 
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
        if args.validation and val_config is not None:
            print(f"\nLoading validation data...")
            val_ds = WaterDataset(
                val_config, 
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

        # 设置保存路径
        model_dir = args.model_dir if args.model_dir else os.path.join(args.output_dir, 'checkpoints', args.model_name)
        log_dir = args.log_dir if args.log_dir else os.path.join(args.output_dir, 'logs')
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 日志记录
        logger = pl_loggers.TensorBoardLogger(log_dir, args.model_name)
        logger.log_hyperparams(vars(args))

        # 创建CSV记录回调（实时写入，启用调试模式）
        log_filename = f"{args.log_name}.csv"
        log_save_path = os.path.join(log_dir, log_filename)
        metrics_callback = MetricsCallback(log_save_path, model_info, args, debug=True)
        metrics_callback.save_model_info()

        # 回调函数配置
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
            
            # 最佳模型保存（自定义命名）
            callbacks.append(ModelCheckpoint(
                dirpath=model_dir,
                filename=f"{args.model_name}_best",
                save_top_k=1,
                monitor=args.monitor_metric,
                mode=args.monitor_metric_mode,
                verbose=True,
                save_last=False,  # 我们单独控制last模型保存
            ))
            
            # 最终模型保存（保存last，使用自定义命名）
            callbacks.append(ModelCheckpoint(
                dirpath=model_dir,
                filename=f"{args.model_name}_last",
                monitor=None,  # 不监控，只保存最后
                save_top_k=0,  # 不基于指标保存
                every_n_epochs=1,  # 每个epoch都检查（实际只保存最后）
                save_on_train_epoch_end=True,  # 在训练epoch结束时保存
                verbose=False,
            ))
            
            # 分步保存（如果指定了interval）
            if args.save_interval > 0:
                callbacks.append(IntervalCheckpoint(
                    save_dir=model_dir,
                    filename_prefix=args.model_name,
                    interval=args.save_interval
                ))
                print(f"分步保存启用: 每 {args.save_interval} 个epoch保存中间模型到 {model_dir}")
            
            # 模型导出（可选）
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
        print("\n" + "="*60)
        print(f"Output Configuration:")
        print(f"  Model Directory: {model_dir}")
        print(f"  Log Directory:   {log_dir}")
        print(f"  Model Prefix:    {args.model_name}")
        print(f"  Save Interval:   {args.save_interval} (0 = no intermediate saves)")
        print(f"  Best Model:      {args.model_name}_best.ckpt")
        print(f"  Last Model:      {args.model_name}_last.ckpt")
        print(f"Accelerator: {accelerator}, Devices: {devices}, Strategy: {strategy}")
        print("="*60 + "\n")
        
        trainer.fit(model, train_dl, val_dl)
        
        print("\nTraining completed!")
        print(f"Models saved to: {model_dir}")
        print(f"Logs saved to:   {log_save_path}")
        
    finally:
        # 清理临时配置文件
        if temp_train_config and os.path.exists(temp_train_config):
            os.remove(temp_train_config)
            print(f"\nCleaned up temp config: {temp_train_config}")
        if temp_val_config and os.path.exists(temp_val_config):
            os.remove(temp_val_config)
            print(f"Cleaned up temp config: {temp_val_config}")


def main():
    args = get_arguments()
    train_water(args)


if __name__ == '__main__':
    main()