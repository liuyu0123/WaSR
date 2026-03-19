################################ 官方模版 ##############################
#模型训练
export CUDA_VISIBLE_DEVICES=0 # GPUs to use(linux)
$env:CUDA_VISIBLE_DEVICES="0"  #(windows)
python train.py `
    --train_config configs/mastr1325_train.yaml `
    --val_config configs/mastr1325_val.yaml `
    --model_name my_wasr `
    --validation `
    --batch_size 4 `
    --epochs 50

#模型预测(对于非IMU版本的模型，无需IMU掩码)
# export CUDA_VISIBLE_DEVICES=-1 # CPU only
export CUDA_VISIBLE_DEVICES=0 # GPU to use
python predict.py `
    --image_dir examples/images `
    --architecture wasr_resnet101 `
    --weights path/to/model/weights.pth `
    --output_dir output/predictions


################################ 水域分割(训练) ##############################
#模型训练(水域分割)
python train_water.py `
    --train_config configs/waterseg_train.yaml `
    --val_config configs/waterseg_val.yaml `
    --model_name water_baseline `
    --batch_size 4 `
    --epochs 5


#适当提高学习率,加速训练
python train_water.py `
    --train_config configs/waterseg_train.yaml `
    --val_config configs/waterseg_val.yaml `
    --model_name water_baseline `
    --batch_size 8 `
    --epochs 5 `
    --precision 16 `
    --learning_rate 1e-4

python train_water.py `
    --train_config configs/waterseg_train.yaml `
    --val_config configs/waterseg_val.yaml `
    --model_name water_fixed `
    --batch_size 8 `
    --epochs 5 `
    --precision 16 `
    --learning_rate 1e-4 `
    --monitor_metric val/iou/water `
    --patience 10

#加速训练
# 综合加速方案(--model)
# --model wasr_resnet50 `
# --model deeplab `
python train_water.py `
    --train_config configs/waterseg_train.yaml `
    --val_config configs/waterseg_val.yaml `
    --model_name water_fast_test `
    --model wasr_resnet50 `
    --batch_size 16 `
    --epochs 20 `
    --precision 16 `
    --learning_rate 1e-3 `
    --monitor_metric val/iou/water `
    --workers 8 `
    --freeze_backbone `
    --no_augmentation

#模型训练-加速版(需 train_water.py 脚本配合修改)
python train_water.py `
    --train_config configs/waterseg_train.yaml `
    --val_config configs/waterseg_val.yaml `
    --model_name water_test_small `
    --model wasr_resnet50 `
    --batch_size 4 `
    --epochs 5 `
    --precision 16 `
    --learning_rate 1e-3 `
    --monitor_metric val/iou/water `
    --workers 1 `
    --no_augmentation

#✅模型训练-加速版(需 train_water.py 脚本配合修改)-修复版
python train_water.py `
    --train_config configs/IRWSB_train.yaml `
    --val_config configs/IRWSB_val.yaml `
    --model_name water_test_stable `
    --model wasr_resnet50 `
    --batch_size 4 `
    --epochs 10 `
    --precision 16 `
    --learning_rate 1e-4 `
    --monitor_metric val/iou/water `
    --workers 1 `
    --water_class_weight 5.0 `
    --freeze_backbone

################################ 水域分割(训练诊断) ##############################
#训练问题诊断
python debug_training.py

################################ 水域分割(模型检测) ##############################


################################ 水域分割(推理) ##############################
#模型预测(水域分割)
export CUDA_VISIBLE_DEVICES=0 # GPU to use(linux)
$env:CUDA_VISIBLE_DEVICES="0"  #(windows)
python predict_water.py `
    --image_dir D:/Files/GitProject/BiSeNet-ooooverflow-LY/dataset/water_seg2/test/images `
    --architecture wasr_resnet50 `
    --weights output_water/logs/water_test_stable/version_0/weights.pth `
    --output_dir output/predictions `
    --overlay_output_dir output/predictions_mask `
    --num_classes 2 `
    --fp16


################################ 水域分割(推理诊断) ##############################
#模型预测数据检查
python check_inference.py
