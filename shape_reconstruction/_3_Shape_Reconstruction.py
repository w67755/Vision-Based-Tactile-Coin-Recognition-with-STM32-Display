import cv2
import yaml
import numpy as np
from collections import deque
from pathlib import Path
from datetime import datetime
from shape_reconstruction import Sensor, Visualizer


# 显示用去噪参数：与硬币分类特征提取所用阈值保持一致。
DISPLAY_CONTACT_THRESHOLD = 0.08
DISPLAY_MEDIAN_KERNEL = 3
DISPLAY_MIN_COMPONENT_AREA = 20
DISPLAY_TEMPORAL_WINDOW = 5  # 约 0.17 秒（相机 30 FPS），仅用于实时显示去闪烁。


def clean_height_map_for_display(height_map):
    """仅清理显示用高度图；原始高度图仍用于保存，避免改动训练数据。"""
    source = np.asarray(height_map, dtype=np.float32)
    valid = np.isfinite(source)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)

    # 3×3 中值滤波去掉孤立亮点，核保持较小以免抹掉细小浮雕。
    filtered = cv2.medianBlur(source, DISPLAY_MEDIAN_KERNEL)
    contact_mask = (valid & (filtered > DISPLAY_CONTACT_THRESHOLD)).astype(np.uint8)

    # 删除面积过小的孤立接触区域，减少 3D 点云中的散点。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        contact_mask, connectivity=8
    )
    clean_mask = np.zeros_like(contact_mask)
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] >= DISPLAY_MIN_COMPONENT_AREA:
            clean_mask[labels == component_id] = 1

    # 仅把干净的接触区域送去显示；非接触区设为零。
    filtered[clean_mask == 0] = 0.0
    contours_result = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_result[-2]
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    return filtered, largest_contour


if __name__ == '__main__':
    f = open("shape_config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    sensor = Sensor(cfg)
    visualizer = Visualizer(sensor.points)

    # 先选择数据用途，验证集单独保存，避免进入训练脚本读取的目录。
    dataset_types = {
        "1": ("shape_dataset", "训练集"),
        "2": ("shape_validation", "验证集"),
    }
    print("请选择本轮数据用途：")
    print("  1: 训练集")
    print("  2: 验证集（新样本预测，不参与训练）")
    while True:
        dataset_choice = input("输入 1 或 2: ").strip()
        if dataset_choice in dataset_types:
            dataset_folder, dataset_name = dataset_types[dataset_choice]
            break
        print("输入无效，请输入 1 或 2。")

    # 每次启动时选择本轮采集的面额类别。
    shape_labels = {
        "1": "1角",
        "2": "1元",
        "3": "5角",
    }
    print("请选择本轮采集的面额类别：")
    for number, label in shape_labels.items():
        print(f"  {number}: {label}")
    while True:
        choice = input("输入类别编号 (1-3): ").strip()
        if choice in shape_labels:
            shape_label = shape_labels[choice]
            break
        print("输入无效，请输入 1、2、3 。")

    # 验证集与训练集分开存放；类别目录不存在时自动创建。
    save_dir = Path("results") / dataset_folder / shape_label
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"本轮{dataset_name}数据将保存到：{save_dir.resolve()}")
    print("按 s 保存当前帧，按 q 退出。")

    # 保存最近几帧的去噪高度图，用逐像素中值减轻传感器噪声造成的闪烁。
    # 该缓存只影响显示，不参与 .npy 原始数据保存。
    display_height_history = deque(maxlen=DISPLAY_TEMPORAL_WINDOW)

    while sensor.cap.isOpened():
        img = sensor.get_rectify_crop_image()
        if img is None:
            continue
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('RawImage_GRAY', img_GRAY)
        # 保留原始高度图用于保存；去噪版本只用于实时深度图和 3D 显示。
        raw_height_map = sensor.raw_image_2_height_map(img_GRAY)
        current_display_map, current_contour = clean_height_map_for_display(raw_height_map)

        if current_contour is None:
            # 松开硬币/没有有效接触时清空历史，避免上一帧形状残留成拖影。
            display_height_history.clear()
            display_height_map = current_display_map
            coin_contour = None
        else:
            display_height_history.append(current_display_map)
            # 时间中值只保留多帧持续存在的起伏，可抑制瞬时亮点和轮廓跳动。
            temporal_map = np.median(
                np.stack(tuple(display_height_history), axis=0), axis=0
            ).astype(np.float32)
            display_height_map, coin_contour = clean_height_map_for_display(temporal_map)

        depth_map = sensor.height_map_2_depth_map(display_height_map)
        depth_map_with_contour = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        if coin_contour is not None:
            # 在二维深度图上用红线标出最大接触轮廓。
            cv2.drawContours(depth_map_with_contour, [coin_contour], -1, (0, 0, 255), 1)
        cv2.imshow('DepthMap', depth_map_with_contour)

        raw_height_map_expanded = sensor.expand_image(raw_height_map)
        display_height_map_expanded = sensor.expand_image(display_height_map)
        key = cv2.waitKey(1)

        # 按 s 保存当前帧：灰度图、显示用深度图，以及可供后续分析的高度矩阵。
        # 请在 RawImage_GRAY 或 DepthMap 窗口获得焦点时按 s。
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(save_dir / f"{timestamp}_gray.png"), img_GRAY)
            cv2.imwrite(str(save_dir / f"{timestamp}_depth.png"), depth_map_with_contour)
            # .npy 仍保存未经本次显示去噪的原始高度图，保持后续分析/训练输入。
            np.save(save_dir / f"{timestamp}_height.npy", raw_height_map_expanded)
            print(f"已保存{dataset_name}类别“{shape_label}”样本：{timestamp}")

        # 按 q 退出程序。
        if key == ord('q'):
            break
        if not visualizer.vis.poll_events():
            break
        else:
            points, gradients = sensor.height_map_2_point_cloud_gradients(
                display_height_map_expanded)
            visualizer.update(points, gradients)
