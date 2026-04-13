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

#模型训练（pro 版）
#方式1
# patience 参数表示多少轮验证集的指标没有提升就停止训练
python train_water_val_pro.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white_noSuffix `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white_noSuffix `
    --model wasr_resnet50 `
    --epochs 10 `
    --batch-size 4 `
    --learning-rate 1e-4 `
    --precision 16 `
    --model-dir checkpoints/experiment1 `
    --log-dir logs/experiment1 `
    --model-name experiment1 `
    --log-name experiment1 `
    --save-interval 0 `
    --water_class_weight 5.0 `
    --freeze_backbone `
    --patience 150

#方式2（兼容原方式）
python train_water_val_pro.py `
    --train_config configs/IRWSB_train.yaml `
    --val_config configs/IRWSB_val.yaml `
    --model_name water_test_stable `
    --model wasr_resnet50 `
    --batch_size 4 `
    --epochs 10 `
    --precision 16 `
    --learning_rate 1e-4 `
    --model-dir checkpoints/experiment1 `
    --save-interval 0

################################ 水域分割(训练诊断) ##############################
#训练问题诊断
python debug_training.py

################################ 水域分割(模型检测) ##############################
#自动生成文件名（默认）
# 输出将保存到 output_water/test_results/test_weights_YYYYMMDD_HHMMSS.csv
python test_water.py `
    --model-path output_water/logs/water_test_stable/version_0/weights.pth `
    --test-images D:\Files\Data\IRWSB\test\images `
    --test-masks D:\Files\Data\IRWSB\test\masks_white_noSuffix `
    --model wasr_resnet50 `
    --batch-size 4 `
    --workers 1
#指定保存csv文件名
python test_water.py `
    --model-path output_water/logs/water_test_stable/version_0/weights.pth `
    --test-images D:\Files\Data\IRWSB\test\images `
    --test-masks D:\Files\Data\IRWSB\test\masks_white_noSuffix `
    --model wasr_resnet50 `
    --batch-size 4 `
    --workers 1 `
    --output-path output_water\test_result.csv


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

#模型推理pro版（生成红色mask蒙版和csv评价指标）
# 单张图片推理 + 保存叠加结果
python predict_water_pro.py `
    --architecture wasr_resnet50 `
    --input ./test.jpg `
    --weights ./model.pth `
    --output ./output/predictions_pro

# 文件夹批量推理 + 真值评估（保存CSV）
# batch_size 设为1，否则所有推理速度相同，fps值完全一致。
python predict_water_pro.py `
    --architecture wasr_resnet50 `
    --input "D:\Files\Data\IRWSB\analyse\images" `
    --weights "F:\AAA\10_wasr_best\experiment1\weights.pth" `
    --gt_mask_dir "D:\Files\Data\IRWSB\analyse\masks_white_noSuffix" `
    --output ./output/predictions_pro `
    --batch_size 1

# 仅评估不打分（终端打印报告）
python predict_water_pro.py `
    --architecture wasr_resnet50 `
    --input "D:\Files\Data\IRWSB\analyse\images" `
    --weights output_water/logs/water_test_stable/version_0/weights.pth `
    --gt_mask_dir "D:\Files\Data\IRWSB\analyse\masks_white_noSuffix"


################################ 水域分割(推理诊断) ##############################
#模型预测数据检查
python check_inference.py
