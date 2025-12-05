#!/usr/bin/env python3
"""
YOLOv8 分割模型训练脚本

用法:
  
  conda activate yolo_training

  # 运行训练
  cd /home/easter/GIP_SAM_YOLO_Calibration
  python train_yolo_seg.py
"""

from ultralytics import YOLO

def main():
    # 从上次训练中断处恢复 (或使用预训练模型开始新训练)
    resume_path = "/home/easter/GIP_SAM_YOLO_Calibration/runs/segment/tactile_seg/weights/last.pt"
    pretrained_model = "/home/easter/GIP_SAM_YOLO_Calibration/models/yolov8s-seg.pt"
    import os
    if os.path.exists(resume_path):
        print(f"恢复训练: {resume_path}")
        model = YOLO(resume_path)
    else:
        print("开始新训练")
        if not os.path.exists(pretrained_model):
            print(f"下载预训练模型到: {pretrained_model}")
            model = YOLO("yolov8s-seg.pt")  # 会自动下载
            # 保存到项目目录
            import shutil
            cache_path = os.path.expanduser("~/.cache/ultralytics")
            for root, dirs, files in os.walk(cache_path):
                if "yolov8s-seg.pt" in files:
                    shutil.copy(os.path.join(root, "yolov8s-seg.pt"), pretrained_model)
                    break
        else:
            model = YOLO(pretrained_model)

    # 训练参数
    results = model.train(
        data="/home/easter/GIP_SAM_YOLO_Calibration/dataset/data.yaml",
        epochs=500,           # 训练轮数
        imgsz=640,            # 图像大小
        batch=32,              # batch size (根据显存调整)
        device=0,             # GPU 设备
        workers=4,            # 数据加载线程数
        patience=50,          # 早停耐心值
        save=True,            # 保存模型
        project="runs/segment",  # 保存目录
        name="tactile_seg",   # 实验名称
        exist_ok=False,       # 不覆盖，自动创建新目录 (tactile_seg, tactile_seg2...)
        pretrained=True,      # 使用预训练权重
        optimizer="auto",     # 优化器
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率 (lr0 * lrf)
        augment=True,         # 数据增强
        amp=False,            # 禁用 AMP 以跳过网络检查
        resume=False,         # 恢复训练
    )

    print("\n训练完成!")
    # 获取实际保存路径
    save_dir = model.trainer.save_dir
    print(f"模型保存目录: {save_dir}")
    print(f"最佳模型路径: {save_dir}/weights/best.pt")
    print(f"\n请将最佳模型复制到 GIP_ws 项目中:")
    print(f"cp {save_dir}/weights/best.pt /home/easter/GIP_ws/src/perception/best.pt")

    # 验证模型
    print("\n验证模型性能...")
    metrics = model.val()
    print(f"mAP50: {metrics.seg.map50:.4f}")
    print(f"mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
