import cv2
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from shape_reconstruction import Sensor, Visualizer


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

    while sensor.cap.isOpened():
        img = sensor.get_rectify_crop_image()
        img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('RawImage_GRAY', img_GRAY)
        height_map = sensor.raw_image_2_height_map(img_GRAY)
        depth_map = sensor.height_map_2_depth_map(height_map)
        cv2.imshow('DepthMap', depth_map)
        height_map = sensor.expand_image(height_map)
        key = cv2.waitKey(1)

        # 按 s 保存当前帧：灰度图、显示用深度图，以及可供后续分析的高度矩阵。
        # 请在 RawImage_GRAY 或 DepthMap 窗口获得焦点时按 s。
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(save_dir / f"{timestamp}_gray.png"), img_GRAY)
            cv2.imwrite(str(save_dir / f"{timestamp}_depth.png"), depth_map)
            np.save(save_dir / f"{timestamp}_height.npy", height_map)
            print(f"已保存{dataset_name}类别“{shape_label}”样本：{timestamp}")

        # 按 q 退出程序。
        if key == ord('q'):
            break
        if not visualizer.vis.poll_events():
            break
        else:
            points, gradients = sensor.height_map_2_point_cloud_gradients(
                height_map)
            visualizer.update(points, gradients)
