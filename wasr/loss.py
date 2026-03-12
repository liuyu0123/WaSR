import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

def water_obstacle_separation_loss(features, gt_mask):
    """Computes the water-obstacle separation loss from intermediate features.
    Args:
        features (torch.tensor): Features tensor
        gt_mask (torch.tensor): Ground truth tensor (可以是 [B, H, W] 或 [B, C, H, W])
    """
    epsilon_watercost = 0.01
    min_samples = 5

    # Resize gt mask to match the extracted features shape (x,y)
    # 针对索引格式的处理
    if gt_mask.dim() == 3:
        # [B, H, W] -> 增加 channel 维度以便 interpolate
        gt_mask_resized = F.interpolate(gt_mask.unsqueeze(1).float(), size=(features.size(2), features.size(3)), mode='nearest')
        # 生成 one-hot 格式的 mask
        mask_water = (gt_mask_resized == 1).float()
        mask_obstacles = (gt_mask_resized == 0).float()
    else:
        # 原有逻辑: [B, C, H, W]
        gt_mask = F.interpolate(gt_mask, size=(features.size(2), features.size(3)), mode='area')
        mask_water = gt_mask[:,1].unsqueeze(1)
        mask_obstacles = gt_mask[:,0].unsqueeze(1)

    # Count number of water and obstacle pixels, clamp to at least 1 (for numerical stability)
    elements_water = mask_water.sum((0,2,3), keepdim=True).clamp(min=1.)
    elements_obstacles = mask_obstacles.sum((0,2,3), keepdim=True)

    # Zero loss if number of samples for any class is smaller than min_samples
    if elements_obstacles.squeeze() < min_samples or elements_water.squeeze() < min_samples:
        return torch.tensor(0., device=features.device)

    # Only keep water and obstacle pixels. Set the rest to 0.
    water_pixels = mask_water * features
    obstacle_pixels = mask_obstacles * features

    # Mean value of water pixels per feature (batch average)
    mean_water = water_pixels.sum((0,2,3), keepdim=True) / elements_water

    # Mean water value matrices for water and obstacle pixels
    mean_water_wat = mean_water * mask_water
    mean_water_obs = mean_water * mask_obstacles

    # Variance of water pixels (per channel, batch average)
    var_water = (water_pixels - mean_water_wat).pow(2).sum((0,2,3), keepdim=True) / elements_water

    # Average quare difference of obstacle pixels and mean water values (per channel)
    difference_obs_wat = (obstacle_pixels - mean_water_obs).pow(2).sum((0,2,3), keepdim=True)

    # Compute the separation loss
    loss_c = elements_obstacles * var_water / (difference_obs_wat + epsilon_watercost)
    var_cost = loss_c.mean()
    return var_cost

def focal_loss(logits, labels, gamma=2.0, alpha=0.25, target_scale='labels', weight=None):
    """Focal loss of the segmentation output `logits` and ground truth `labels`.
    
    Args:
        logits: [B, C, H, W]
        labels: [B, H, W] (Class indices) or [B, C, H, W] (One-hot)
        gamma: focal loss gamma
        alpha: focal loss alpha (used if weight is None)
        target_scale: 'logits' or 'labels'
        weight: Tensor [C] class weights
    """
    epsilon = 1.e-9
    num_classes = logits.size(1)

    # 尺寸对齐
    if target_scale == 'logits':
        logits_size = (logits.size(2), logits.size(3))
        if labels.dim() == 3:
            labels = F.interpolate(labels.unsqueeze(1).float(), size=logits_size, mode='nearest').squeeze(1)
        else:
            labels = F.interpolate(labels, size=logits_size, mode='area')
    elif target_scale == 'labels':
        labels_size = (labels.size(1), labels.size(2)) if labels.dim() == 3 else (labels.size(2), labels.size(3))
        logits = TF.resize(logits, labels_size, interpolation=InterpolationMode.BILINEAR)
    else:
        raise ValueError('Invalid value for target_scale: %s' % target_scale)

    logits_sm = torch.softmax(logits, 1)
    
    # === 核心修改：支持 [B, H, W] 索引格式 ===
    if labels.dim() == 3: 
        # 输入是 [B, H, W] 类别索引
        # 1. 计算 CrossEntropy 风格的 Focal Loss
        # 获取真实类别的概率
        # labels: [B, H, W] -> [B, 1, H, W] 用于 gather
        labels_index = labels.unsqueeze(1) 
        p_t = logits_sm.gather(1, labels_index).squeeze(1) # [B, H, W]
        
        # 2. 计算 Focal 权重
        focal_factor = (1 - p_t) ** gamma
        
        # 3. 计算权重
        if weight is not None:
            # 根据每个像素的类别索引取出对应的权重
            class_weights = weight[labels] # [B, H, W]
        else:
            class_weights = alpha
            
        # 4. 最终 Loss
        # - weight * focal_factor * log(p_t)
        # 使用 clamp 防止 log(0)
        loss = -class_weights * focal_factor * torch.log(p_t.clamp(min=epsilon))
        
    else:
        # 输入是 [B, C, H, W] One-hot (原始逻辑)
        # 兼容旧代码
        fl = -labels * torch.log(logits_sm + epsilon) * (1. - logits_sm) ** gamma
        
        if weight is not None:
            # weight: [C] -> [1, C, 1, 1]
            weight_map = weight.view(1, -1, 1, 1)
            fl = fl * weight_map
            
        loss = fl.sum(1) # [B, H, W]

    return loss.mean()
