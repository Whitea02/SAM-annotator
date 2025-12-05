#!/usr/bin/env python3
"""
SAM2 辅助标注工具 - 用于视触觉传感器图像分割标注
功能:
  1. 采集模式: 从摄像头采集图像
  2. 标注模式: 使用 SAM2 辅助分割标注
  3. 导出 YOLO 格式数据集

用法:
  # 激活环境
  conda activate sam3

  # 采集图像
  python annotator.py collect --camera 0 --output ./dataset/images

  # 标注图像 (使用 SAM2 辅助)
  python annotator.py annotate --input ./dataset/images --output ./dataset

  # 查看已标注数据
  python annotator.py view --input ./dataset

快捷键:
  采集模式:
    Space - 保存当前帧
    q - 退出

  标注模式:
    m - 切换 SAM2/手动 标注模式

    SAM2模式:
      左键点击 - 正样本点 (前景)
      右键点击 - 负样本点 (背景)

    手动模式:
      左键点击 - 添加多边形顶点
      c - 完成多边形并保存

    通用:
      0-9 - 选择类别
      s - 保存当前标注 (SAM2模式)
      r - 删除 dataset 中对应的 image 和 labels 文件
      d - 删除当前图片所有标注
      n - 下一张图像
      p - 上一张图像
      q - 退出
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil

# SAM2 imports
import torch


class ImageCollector:
    """图像采集器"""

    def __init__(self, camera_id: int, output_dir: str):
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = self._get_start_index()

    def _get_start_index(self) -> int:
        """获取起始编号,避免覆盖已有图像"""
        existing = list(self.output_dir.glob("*.jpg")) + list(self.output_dir.glob("*.png"))
        if not existing:
            return 0
        indices = []
        for f in existing:
            try:
                idx = int(f.stem.split('_')[-1])
                indices.append(idx)
            except:
                pass
        return max(indices) + 1 if indices else 0

    def run(self):
        """运行采集"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_id}")
            return

        print(f"Camera opened. Saving to: {self.output_dir}")
        print("Press SPACE to save frame, 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # 显示帧计数
            display = frame.copy()
            cv2.putText(display, f"Saved: {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "SPACE: save | q: quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Collector', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                filename = f"img_{self.frame_count:05d}.jpg"
                filepath = self.output_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"Saved: {filepath}")
                self.frame_count += 1
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Collection finished. Total images: {self.frame_count}")


class SAM2Annotator:
    """SAM2 辅助标注器"""

    def __init__(self, input_dir: str, output_dir: str, class_names: list):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_names = class_names

        # 创建输出目录
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 获取图像列表
        self.image_files = sorted(
            list(self.input_dir.glob("*.jpg")) +
            list(self.input_dir.glob("*.png"))
        )

        if not self.image_files:
            print(f"No images found in {self.input_dir}")
            sys.exit(1)

        print(f"Found {len(self.image_files)} images")

        # 当前状态
        self.current_idx = 0
        self.current_class = 0
        self.annotation_mode = "sam"  # "sam" 或 "manual"
        self.positive_points = []  # 前景点
        self.negative_points = []  # 背景点
        self.manual_polygon_points = []  # 手动多边形顶点
        self.current_mask = None
        self.annotations = []  # 当前图像的所有标注

        # 加载 SAM2 模型
        self._load_sam2()

    def _load_sam2(self):
        """加载 SAM2 模型"""
        print("Loading SAM2 model...")
        try:
            # Fix for SAM2 import issue:
            # Add the sam2 repository root to sys.path so we import the inner 'sam2' package
            sam2_repo_path = os.path.join(os.getcwd(), "sam2")
            if sam2_repo_path not in sys.path:
                sys.path.insert(0, sam2_repo_path)

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # 查找 checkpoint 和 config (使用绝对路径避免目录名冲突)
            checkpoint = Path("/home/easter/GIP_SAM_YOLO_Calibration/sam2/checkpoints/sam2.1_hiera_small.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

            if not checkpoint.exists():
                print(f"Checkpoint not found: {checkpoint}")
                print("Please download the checkpoint first")
                self.predictor = None
                return

            # 构建模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)

            print("SAM2 model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load SAM2 model: {e}")
            import traceback
            traceback.print_exc()
            print("Running in manual annotation mode (no SAM2 assistance)")
            self.predictor = None

    def _set_image(self, image_path: str):
        """设置当前图像"""
        self.current_image = cv2.imread(str(image_path))
        self.image_height, self.image_width = self.current_image.shape[:2]

        if self.predictor is not None:
            # 为 SAM2 准备图像 (需要 RGB)
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(image_rgb)

        # 重置点和标注
        self.positive_points = []
        self.negative_points = []
        self.manual_polygon_points = []
        self.current_mask = None
        self.annotations = []

        # 加载已有标注
        self._load_existing_annotations()

    def _load_existing_annotations(self):
        """加载已有标注"""
        label_file = self.labels_dir / f"{self.image_files[self.current_idx].stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        points = [float(x) for x in parts[1:]]
                        # 转换为像素坐标
                        pixel_points = []
                        for i in range(0, len(points), 2):
                            x = int(points[i] * self.image_width)
                            y = int(points[i+1] * self.image_height)
                            pixel_points.append((x, y))
                        self.annotations.append({
                            'class_id': class_id,
                            'polygon': pixel_points
                        })

    def _predict_mask(self):
        """使用 SAM2 预测 mask"""
        if self.predictor is None:
            return

        if not self.positive_points and not self.negative_points:
            self.current_mask = None
            return

        try:
            # 准备点坐标和标签
            all_points = []
            all_labels = []

            for px, py in self.positive_points:
                all_points.append([px, py])
                all_labels.append(1)  # 前景

            for px, py in self.negative_points:
                all_points.append([px, py])
                all_labels.append(0)  # 背景

            point_coords = np.array(all_points)
            point_labels = np.array(all_labels)

            # 预测
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

            # 选择得分最高的 mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            # 转换为 uint8
            self.current_mask = (mask * 255).astype(np.uint8)

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            self.current_mask = None

    def _get_polygon_from_mask(self, mask):
        """从 mask 提取多边形轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 取最大轮廓
        largest = max(contours, key=cv2.contourArea)

        # 简化轮廓
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        # 转换为点列表
        polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]
        return polygon if len(polygon) >= 3 else None

    def _save_annotation(self):
        """保存当前标注为 YOLO 格式"""
        if self.current_mask is None:
            print("No mask to save")
            return

        polygon = self._get_polygon_from_mask(self.current_mask)
        if polygon is None:
            print("Failed to extract polygon from mask")
            return

        # 添加到标注列表
        self.annotations.append({
            'class_id': self.current_class,
            'polygon': polygon
        })

        # 保存到文件
        self._write_label_file()

        # 复制图像到输出目录
        src_image = self.image_files[self.current_idx]
        dst_image = self.images_dir / src_image.name
        if not dst_image.exists():
            shutil.copy(str(src_image), str(dst_image))

        print(f"Saved annotation: class={self.class_names[self.current_class]}, points={len(polygon)}")

        # 重置当前标注状态
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None

    def _save_manual_polygon(self):
        """保存手动绘制的多边形"""
        if len(self.manual_polygon_points) < 3:
            print("Need at least 3 points to form a polygon")
            return

        # 添加到标注列表
        self.annotations.append({
            'class_id': self.current_class,
            'polygon': self.manual_polygon_points.copy()
        })

        # 保存到文件
        self._write_label_file()

        # 复制图像到输出目录
        src_image = self.image_files[self.current_idx]
        dst_image = self.images_dir / src_image.name
        if not dst_image.exists():
            shutil.copy(str(src_image), str(dst_image))

        print(f"Saved manual polygon: class={self.class_names[self.current_class]}, points={len(self.manual_polygon_points)}")

        # 重置手动多边形
        self.manual_polygon_points = []

    def _delete_all_annotations(self):
        """删除当前图片的所有标注"""
        # 清空内存中的标注
        self.annotations = []

        # 删除标签文件
        label_file = self.labels_dir / f"{self.image_files[self.current_idx].stem}.txt"
        if label_file.exists():
            label_file.unlink()
            print(f"Deleted all annotations for current image")
        else:
            print("No annotations to delete")

        # 重置当前标注状态
        self.positive_points = []
        self.negative_points = []
        self.manual_polygon_points = []
        self.current_mask = None

    def _delete_current_files(self):
        """删除 dataset 文件夹中对应的 image 和 labels 文件"""
        src_image = self.image_files[self.current_idx]

        # 删除 dataset/images 中的图片
        dst_image = self.images_dir / src_image.name
        if dst_image.exists():
            dst_image.unlink()
            print(f"Deleted image: {dst_image}")

        # 删除 dataset/labels 中的标签
        label_file = self.labels_dir / f"{src_image.stem}.txt"
        if label_file.exists():
            label_file.unlink()
            print(f"Deleted label: {label_file}")

        if not dst_image.exists() and not label_file.exists():
            print("No saved files to delete for this image")

        # 清空内存中的标注
        self.annotations = []

        # 重置当前标注状态
        self.positive_points = []
        self.negative_points = []
        self.manual_polygon_points = []
        self.current_mask = None

    def _write_label_file(self):
        """写入标签文件"""
        label_file = self.labels_dir / f"{self.image_files[self.current_idx].stem}.txt"

        with open(label_file, 'w') as f:
            for ann in self.annotations:
                class_id = ann['class_id']
                polygon = ann['polygon']

                # 转换为归一化坐标
                normalized = []
                for x, y in polygon:
                    normalized.append(f"{x / self.image_width:.6f}")
                    normalized.append(f"{y / self.image_height:.6f}")

                line = f"{class_id} " + " ".join(normalized)
                f.write(line + "\n")

    def _draw_display(self):
        """绘制显示画面"""
        display = self.current_image.copy()

        # 绘制已保存的标注
        for ann in self.annotations:
            polygon = np.array(ann['polygon'], dtype=np.int32)
            color = self._get_class_color(ann['class_id'])
            cv2.polylines(display, [polygon], True, color, 2)
            # 半透明填充
            overlay = display.copy()
            cv2.fillPoly(overlay, [polygon], color)
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # 绘制当前 mask (降低透明度以便看清原图)
        if self.current_mask is not None:
            color = self._get_class_color(self.current_class)
            mask_colored = np.zeros_like(display)
            mask_colored[self.current_mask > 0] = color
            display = cv2.addWeighted(display, 0.95, mask_colored, 0.05, 0)

            # 绘制轮廓
            contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color, 2)

        # 绘制点
        for px, py in self.positive_points:
            cv2.circle(display, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 7, (255, 255, 255), 1)

        for px, py in self.negative_points:
            cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
            cv2.circle(display, (px, py), 7, (255, 255, 255), 1)

        # 绘制手动多边形的点和线
        if self.manual_polygon_points:
            color = self._get_class_color(self.current_class)
            # 绘制已有的边
            for i in range(len(self.manual_polygon_points)):
                pt1 = self.manual_polygon_points[i]
                pt2 = self.manual_polygon_points[(i + 1) % len(self.manual_polygon_points)]
                cv2.line(display, pt1, pt2, color, 2)

            # 绘制顶点
            for px, py in self.manual_polygon_points:
                cv2.circle(display, (px, py), 6, color, -1)
                cv2.circle(display, (px, py), 8, (255, 255, 255), 2)

        # 绘制信息
        mode_text = "SAM2" if self.annotation_mode == "sam" else "MANUAL"
        info_lines = [
            f"Image: {self.current_idx + 1}/{len(self.image_files)}",
            f"File: {self.image_files[self.current_idx].name}",
            f"Mode: {mode_text}",
            f"Class: [{self.current_class}] {self.class_names[self.current_class]}",
            f"Annotations: {len(self.annotations)}",
            "",
            "m: toggle mode | 0-9: class | r: delete files | d: delete annotations",
        ]

        if self.annotation_mode == "sam":
            info_lines.append("L-click: +point | R-click: -point | s: save")
        else:
            info_lines.append(f"L-click: add point ({len(self.manual_polygon_points)}) | c: complete")

        info_lines.append("n/p: next/prev | q: quit")

        y = 30
        for line in info_lines:
            cv2.putText(display, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

        return display

    def _get_class_color(self, class_id: int):
        """获取类别颜色"""
        colors = [
            (0, 255, 0),    # 绿
            (255, 0, 0),    # 蓝
            (0, 0, 255),    # 红
            (255, 255, 0),  # 青
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 黄
            (128, 255, 0),
            (255, 128, 0),
            (128, 0, 255),
            (0, 128, 255),
        ]
        return colors[class_id % len(colors)]

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if self.annotation_mode == "sam":
            # SAM2 模式
            if event == cv2.EVENT_LBUTTONDOWN:
                self.positive_points.append((x, y))
                self._predict_mask()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.negative_points.append((x, y))
                self._predict_mask()
        else:
            # 手动多边形模式
            if event == cv2.EVENT_LBUTTONDOWN:
                self.manual_polygon_points.append((x, y))
                print(f"Added point {len(self.manual_polygon_points)}: ({x}, {y})")

    def run(self):
        """运行标注器"""
        cv2.namedWindow('Annotator')
        cv2.setMouseCallback('Annotator', self._mouse_callback)

        self._set_image(self.image_files[self.current_idx])

        while True:
            display = self._draw_display()
            cv2.imshow('Annotator', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):
                # 下一张
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self._set_image(self.image_files[self.current_idx])
            elif key == ord('p'):
                # 上一张
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self._set_image(self.image_files[self.current_idx])
            elif key == ord('r'):
                # 删除 dataset 文件夹中对应的 image 和 labels 文件
                self._delete_current_files()
            elif key == ord('s'):
                # 保存 (仅SAM模式)
                if self.annotation_mode == "sam":
                    self._save_annotation()
                else:
                    print("Use 'c' to complete polygon in manual mode")
            elif key == ord('m'):
                # 切换模式
                self.annotation_mode = "manual" if self.annotation_mode == "sam" else "sam"
                # 切换模式时重置当前标注
                self.positive_points = []
                self.negative_points = []
                self.manual_polygon_points = []
                self.current_mask = None
                print(f"Switched to {self.annotation_mode.upper()} mode")
            elif key == ord('c'):
                # 完成多边形 (仅手动模式)
                if self.annotation_mode == "manual":
                    self._save_manual_polygon()
                else:
                    print("'c' is only available in manual mode")
            elif key == ord('d'):
                # 删除所有标注
                self._delete_all_annotations()
            elif ord('0') <= key <= ord('9'):
                # 选择类别
                class_id = key - ord('0')
                if class_id < len(self.class_names):
                    self.current_class = class_id
                    print(f"Selected class: [{class_id}] {self.class_names[class_id]}")

        cv2.destroyAllWindows()
        print("Annotation finished")

        # 生成 data.yaml
        self._generate_data_yaml()

    def _generate_data_yaml(self):
        """生成 YOLO 数据集配置文件"""
        yaml_content = f"""# YOLOv8 Segmentation Dataset
# Generated by SAM2 Annotator

path: {self.output_dir.absolute()}
train: images
val: images

names:
"""
        for i, name in enumerate(self.class_names):
            yaml_content += f"  {i}: {name}\n"

        yaml_file = self.output_dir / "data.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)

        print(f"Generated: {yaml_file}")


class DatasetViewer:
    """数据集查看器"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_dir = self.dataset_dir / "labels"

        # 加载类别名称
        yaml_file = self.dataset_dir / "data.yaml"
        self.class_names = ["object"]
        if yaml_file.exists():
            import yaml
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    self.class_names = list(data['names'].values())

        # 获取图像列表
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )

        if not self.image_files:
            print(f"No images found in {self.images_dir}")
            sys.exit(1)

    def run(self):
        """运行查看器"""
        current_idx = 0

        while True:
            image_path = self.image_files[current_idx]
            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]

            # 读取标签
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            points = [float(x) for x in parts[1:]]

                            # 转换为像素坐标
                            polygon = []
                            for i in range(0, len(points), 2):
                                x = int(points[i] * w)
                                y = int(points[i+1] * h)
                                polygon.append([x, y])

                            polygon = np.array(polygon, dtype=np.int32)

                            # 绘制
                            colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
                            color = colors[class_id % len(colors)]

                            cv2.polylines(image, [polygon], True, color, 2)
                            overlay = image.copy()
                            cv2.fillPoly(overlay, [polygon], color)
                            image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

                            # 类别标签
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            cv2.putText(image, class_name, (polygon[0][0], polygon[0][1] - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 显示信息
            cv2.putText(image, f"{current_idx + 1}/{len(self.image_files)}: {image_path.name}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "n: next | p: prev | q: quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Viewer', image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_idx = min(current_idx + 1, len(self.image_files) - 1)
            elif key == ord('p'):
                current_idx = max(current_idx - 1, 0)

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='SAM2 Annotation Tool')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # 采集命令
    collect_parser = subparsers.add_parser('collect', help='Collect images from camera')
    collect_parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    collect_parser.add_argument('--output', type=str, default='./dataset/raw_images',
                               help='Output directory')

    # 标注命令
    annotate_parser = subparsers.add_parser('annotate', help='Annotate images with SAM2')
    annotate_parser.add_argument('--input', type=str, required=True,
                                help='Input images directory')
    annotate_parser.add_argument('--output', type=str, default='./dataset',
                                help='Output dataset directory')
    annotate_parser.add_argument('--classes', type=str, nargs='+',
                                default=['contact'],
                                help='Class names (default: contact for tactile object contour)')

    # 查看命令
    view_parser = subparsers.add_parser('view', help='View annotated dataset')
    view_parser.add_argument('--input', type=str, required=True,
                            help='Dataset directory')

    args = parser.parse_args()

    if args.command == 'collect':
        collector = ImageCollector(args.camera, args.output)
        collector.run()
    elif args.command == 'annotate':
        annotator = SAM2Annotator(args.input, args.output, args.classes)
        annotator.run()
    elif args.command == 'view':
        viewer = DatasetViewer(args.input)
        viewer.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
