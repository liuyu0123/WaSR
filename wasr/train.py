from PIL import Image
import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from .loss import focal_loss, water_obstacle_separation_loss
from .metrics import PixelAccuracy, ClassIoU

NUM_EPOCHS = 50
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
LR_DECAY_POW = 0.9
FOCAL_LOSS_SCALE = 'labels'
SL_LAMBDA = 0.01

class LitModel(pl.LightningModule):
    """ Pytorch Lightning wrapper for a model, ready for distributed training. """
    @staticmethod
    def add_argparse_args(parser):
        """Adds model specific parameters to parser."""
        parser.add_argument("--learning_rate", "--learning-rate", "--lr", type=float, default=LEARNING_RATE, help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum component of the optimiser.")
        parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs.")
        parser.add_argument("--lr_decay_pow", type=float, default=LR_DECAY_POW, help="Decay parameter to compute the learning rate decay.")
        parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--focal_loss_scale", type=str, default=FOCAL_LOSS_SCALE, choices=['logits', 'labels'], help="Which scale to use for focal loss computation (logits or labels).")
        parser.add_argument("--no_separation_loss", action='store_true', help="Disable separation loss.")
        parser.add_argument("--separation_loss_lambda", default=SL_LAMBDA, type=float, help="The separation loss lambda (weight).")
        return parser

    def __init__(self, model, num_classes, args):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.lr_decay_pow = args.lr_decay_pow
        self.focal_loss_scale = args.focal_loss_scale
        self.separation_loss = not args.no_separation_loss
        self.separation_loss_lambda = args.separation_loss_lambda
        
        # 处理类别权重
        self.register_buffer('class_weights', torch.ones(num_classes))
        if hasattr(args, 'water_class_weight') and num_classes >= 2:
            self.class_weights[1] = args.water_class_weight
            print(f"Class weights initialized: {self.class_weights.tolist()}")

        # Metrics - 训练集和验证集分开
        self.train_accuracy = PixelAccuracy(num_classes)
        self.val_accuracy = PixelAccuracy(num_classes)
        
        # IoU 指标
        self.train_iou_0 = ClassIoU(0, num_classes)
        self.train_iou_1 = ClassIoU(1, num_classes)
        self.val_iou_0 = ClassIoU(0, num_classes)
        self.val_iou_1 = ClassIoU(1, num_classes)
        
        if num_classes > 2:
            self.train_iou_2 = ClassIoU(2, num_classes)
            self.val_iou_2 = ClassIoU(2, num_classes)
        
        # 用于计算 Precision, Recall, F1 的混淆矩阵累积
        # 使用 register_buffer 确保它在正确的设备上
        self.register_buffer('train_confusion_matrix', torch.zeros(num_classes, num_classes))
        self.register_buffer('val_confusion_matrix', torch.zeros(num_classes, num_classes))

    def _reset_confusion_matrix(self, stage):
        """重置混淆矩阵"""
        cm = getattr(self, f'{stage}_confusion_matrix')
        cm.zero_()

    def _update_confusion_matrix(self, preds, labels, stage):
        """更新混淆矩阵"""
        # preds 和 labels 都是 [B, H, W] 的索引格式
        cm = getattr(self, f'{stage}_confusion_matrix')
        
        # 确保在同一设备上
        device = cm.device
        preds = preds.to(device)
        labels = labels.to(device)
        
        # 展平
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)
        
        # 只计算有效标签
        valid_mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
        preds_valid = preds_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        # 累积到混淆矩阵
        for t in range(self.num_classes):
            for p in range(self.num_classes):
                cm[t, p] += ((labels_valid == t) & (preds_valid == p)).sum()

    def _compute_metrics_from_cm(self, stage):
        """基于混淆矩阵计算 Precision, Recall, F1"""
        cm = getattr(self, f'{stage}_confusion_matrix')
        
        # 避免除零
        eps = 1e-7
        
        # 对每个类别计算
        precisions = []
        recalls = []
        f1s = []
        ious = []
        
        for c in range(self.num_classes):
            tp = cm[c, c]  # 真正例
            fp = cm[:, c].sum() - tp  # 假正例
            fn = cm[c, :].sum() - tp  # 假反例
            
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            iou = tp / (tp + fp + fn + eps)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            ious.append(iou)
        
        # 返回 water 类（类别1）的指标，以及 mIoU
        return {
            'precision': precisions[1].item() if self.num_classes > 1 else 0,
            'recall': recalls[1].item() if self.num_classes > 1 else 0,
            'f1': f1s[1].item() if self.num_classes > 1 else 0,
            'miou': (sum(ious) / len(ious)).item()  # 平均IoU
        }

    def forward(self, x):
        output = self.model(x)
        return output['out']

    def _compute_metrics(self, logits, labels, stage='train'):
        """计算并记录指标"""
        # Resize logits to match label size
        if labels.dim() == 3:
            # 索引格式 [B, H, W]
            labels_size = (labels.size(1), labels.size(2))
            labels_hard = labels
        else:
            # One-hot 格式 [B, C, H, W]
            labels_size = (labels.size(2), labels.size(3))
            labels_hard = labels.argmax(1)

        logits = TF.resize(logits, labels_size, interpolation=Image.BILINEAR)
        preds = logits.argmax(1)

        # 根据 stage 选择对应的 metrics
        if stage == 'train':
            accuracy_metric = self.train_accuracy
            iou_0_metric = self.train_iou_0
            iou_1_metric = self.train_iou_1
            iou_2_metric = self.train_iou_2 if self.num_classes > 2 else None
        else:
            accuracy_metric = self.val_accuracy
            iou_0_metric = self.val_iou_0
            iou_1_metric = self.val_iou_1
            iou_2_metric = self.val_iou_2 if self.num_classes > 2 else None

        # 更新指标
        accuracy_metric(preds, labels_hard)
        iou_0_metric(preds, labels_hard)
        iou_1_metric(preds, labels_hard)
        
        # 更新混淆矩阵用于计算 Precision, Recall, F1
        self._update_confusion_matrix(preds, labels_hard, stage)
        
        # 记录指标（step级别）
        self.log(f'{stage}/accuracy', accuracy_metric, on_step=(stage=='train'), on_epoch=True)
        self.log(f'{stage}/iou/obstacle', iou_0_metric, on_step=(stage=='train'), on_epoch=True)
        self.log(f'{stage}/iou/water', iou_1_metric, on_step=(stage=='train'), on_epoch=True)
        
        if iou_2_metric is not None:
            iou_2_metric(preds, labels_hard)
            self.log(f'{stage}/iou/sky', iou_2_metric, on_step=(stage=='train'), on_epoch=True)

        return preds

    def training_step(self, batch, batch_idx):
        features, labels = batch
        out = self.model(features)
        
        fl = focal_loss(out['out'], labels['segmentation'], 
                        target_scale=self.focal_loss_scale, 
                        weight=self.class_weights)
        
        if self.separation_loss:
            separation_loss = water_obstacle_separation_loss(out['aux'], labels['segmentation'])
        else:
            separation_loss = torch.tensor(0.0, device=self.device)
            
        separation_loss = self.separation_loss_lambda * separation_loss
        loss = fl + separation_loss

        # 计算并记录训练指标
        with torch.no_grad():
            self._compute_metrics(out['out'], labels['segmentation'], stage='train')

        # log losses
        self.log('train/loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/focal_loss', fl.item(), on_step=True, on_epoch=True)
        self.log('train/separation_loss', separation_loss.item(), on_step=True, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self):
        """训练epoch结束时，计算并记录 Precision, Recall, F1"""
        metrics = self._compute_metrics_from_cm('train')
        
        self.log('train/precision', metrics['precision'], on_epoch=True)
        self.log('train/recall', metrics['recall'], on_epoch=True)
        self.log('train/f1', metrics['f1'], on_epoch=True)
        self.log('train/miou', metrics['miou'], on_epoch=True)
        
        # 重置混淆矩阵
        self._reset_confusion_matrix('train')

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        out = self.model(features)
        
        loss = focal_loss(out['out'], labels['segmentation'], 
                        target_scale=self.focal_loss_scale, 
                        weight=self.class_weights)
        
        # 关键修复：使用 sync_dist=True 确保分布式同步，并明确 on_epoch=True
        self.log('val/loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)

        # 计算并记录验证指标
        self._compute_metrics(out['out'], labels['segmentation'], stage='val')
        
        return {'loss': loss}

    def on_validation_epoch_end(self):
        """验证epoch结束时，计算并记录 Precision, Recall, F1"""
        metrics = self._compute_metrics_from_cm('val')
        
        # 关键修复：确保在epoch结束时记录，使用 sync_dist
        self.log('val/precision', metrics['precision'], on_epoch=True, sync_dist=True)
        self.log('val/recall', metrics['recall'], on_epoch=True, sync_dist=True)
        self.log('val/f1', metrics['f1'], on_epoch=True, sync_dist=True)
        self.log('val/miou', metrics['miou'], on_epoch=True, sync_dist=True)
        
        # 重置混淆矩阵
        self._reset_confusion_matrix('val')
        
        # 可选：打印当前epoch的混淆矩阵用于调试
        if self.trainer.is_global_zero:  # 只在主进程打印
            print(f"\n[DEBUG] Epoch {self.current_epoch} Val Confusion Matrix:\n{self.val_confusion_matrix}")

    def configure_optimizers(self):
        # Separate parameters for different LRs
        encoder_parameters = []
        decoder_w_parameters = []
        decoder_b_parameters = []
        for name, parameter in self.model.named_parameters():
            if name.startswith('backbone'):
                encoder_parameters.append(parameter)
            elif 'weight' in name:
                decoder_w_parameters.append(parameter)
            else:
                decoder_b_parameters.append(parameter)

        optimizer = torch.optim.RMSprop([
            {'params': encoder_parameters, 'lr': self.learning_rate},
            {'params': decoder_w_parameters, 'lr': self.learning_rate * 10},
            {'params': decoder_b_parameters, 'lr': self.learning_rate * 20},
        ], momentum=self.momentum, alpha=0.9, weight_decay=self.weight_decay)

        # Decaying LR function
        lr_fn = lambda epoch: (1 - epoch/self.epochs) ** self.lr_decay_pow
        scheduler = LambdaLR(optimizer, lr_fn)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model'] = self.model.state_dict()