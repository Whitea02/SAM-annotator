# SAM-YOLO Auto Annotator

这是一个结合了 Meta SAM 2 (Segment Anything Model 2) 和 YOLOv8 的自动化数据标注与训练工具。它旨在加速视触觉传感器（或任何视觉任务）的数据集制作和模型训练流程。

## 主要功能

*   **半自动标注**: 利用 SAM 2 的强大分割能力，只需点击几下即可生成高质量的分割掩码。
*   **YOLOv8 训练集成**: 一键将标注数据转换为 YOLO 格式并开始训练。
*   **多边形交互**: 提供简单的交互界面进行样本采集和标注微调。

## 目录结构

```
sam-yolo-annotator/
├── README.md                 # 项目文档
├── sam2_annotator.py         # SAM2 辅助标注工具 (核心)
├── train_yolo_seg.py         # YOLO 训练脚本
├── dataset/                  # 数据集目录
│   ├── data.yaml            # YOLO 数据集配置
│   ├── raw_images/          # 原始采集图像
│   ├── images/              # 训练集图像
│   └── labels/              # 训练集标签 (YOLO txt格式)
├── runs/                     # 训练日志与模型输出
├── sam2/                     # SAM2 库与权重
└── sam3/                     # SAM2 环境配置
```

## 环境安装

本项目依赖两个 Python 环境（避免依赖冲突）：

1.  **标注环境 (`sam3`)**: 用于运行 SAM 2。
2.  **训练环境 (`venv` / `yolo_training`)**: 用于训练 YOLOv8。

### 1. SAM 2 环境配置
请参考 `sam2/README.md` 或官方文档安装 SAM 2 依赖。通常需要 PyTorch 和 CUDA 支持。

```bash
# 示例：创建并激活 conda 环境
conda create -n sam3 python=3.10
conda activate sam3
# 安装依赖...
```

首次使用需下载 SAM 2 权重：
```bash
cd sam2/checkpoints
./download_ckpts.sh
```

### 2. YOLO 训练环境
```bash
# 激活你的 YOLO 环境
source /path/to/venv/bin/activate
# 或
conda activate yolo_training
pip install ultralytics
```

## 使用指南

### 1. 数据采集
使用摄像头采集原始图像：
```bash
conda activate sam3
python sam2_annotator.py collect --camera 0 --output ./dataset/raw_images
```
*   `Space`: 保存图像
*   `q`: 退出

### 2. 交互式标注
利用 SAM 2 进行半自动标注：
```bash
conda activate sam3
python sam2_annotator.py annotate --input ./dataset/raw_images --output ./dataset
```
*   **左键**: 添加前景点（物体）
*   **右键**: 添加背景点（去除区域）
*   **s**: 保存标注
*   **n/p**: 切换图像
*   **r**: 重置当前图

### 3. 模型训练
切换到训练环境并运行训练脚本。每次运行会自动创建新的实验文件夹（如 `tactile_seg2`, `tactile_seg3`...）。

```bash
conda activate yolo_training
python train_yolo_seg.py
```

### 4. 部署模型
训练完成后，脚本会提示最佳模型的保存路径。请手动将其复制到你的应用项目中。

例如：
```bash
# 复制到 GIP_ws 机器人项目
cp runs/segment/tactile_segX/weights/best.pt /home/easter/GIP_ws/src/perception/best.pt
```

## 许可证
[MIT License](LICENSE)