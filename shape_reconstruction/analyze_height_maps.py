"""批量检查已保存的高度矩阵，并计算触觉接触特征。

运行方式：在 PyCharm 中运行本文件。程序会递归读取
results/shape_dataset/<类别>/ 下的 *_height.npy，并生成带类别标签的 CSV。
"""

import csv
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SHAPE_DATASET_DIR = RESULTS_DIR / "shape_dataset"
SCALE_PATH = (
    BASE_DIR / "calibration" / "sensor_1" / "camera_calibration"
    / "position_scale.npy"
)

# 与项目 3D 可视化中的非接触点阈值一致，先作为初始接触阈值。
# 应采集未按压数据后检查噪声，再根据数据调整这个值。
CONTACT_THRESHOLD = 0.08


def profile_statistics(profile_values, profile_positions):
    """计算一条高度剖面的起伏统计；Ra/Rq 仅为相对剖面指标。"""
    if len(profile_values) == 0:
        return {
            "count": 0,
            "mean": "",
            "peak_to_valley": "",
            "ra": "",
            "rq": "",
        }

    values = np.asarray(profile_values, dtype=np.float64)
    positions = np.asarray(profile_positions, dtype=np.float64)

    # 用直线拟合去除剖面整体倾斜，再统计剩余起伏。
    if len(values) >= 2 and np.ptp(positions) > 0:
        slope, intercept = np.polyfit(positions, values, 1)
        residual = values - (slope * positions + intercept)
    else:
        residual = values - values.mean()

    return {
        "count": int(len(values)),
        "mean": round(float(values.mean()), 6),
        "peak_to_valley": round(float(values.max() - values.min()), 6),
        # 下列 Ra/Rq 是当前高度矩阵单位下的相对值，未确认标定前不是微米或毫米。
        "ra": round(float(np.mean(np.abs(residual))), 6),
        "rq": round(float(np.sqrt(np.mean(residual ** 2))), 6),
    }


def extract_contour_features(contact_mask, height_map, mm_per_pixel=None):
    """提取最大接触轮廓特征，并取穿过接触区的横向、纵向高度剖面。"""
    features = {
        "contact_components": 0,
        "contour_area_px": "",
        "contour_perimeter_px": "",
        "bbox_width_px": "",
        "bbox_height_px": "",
        "bbox_aspect_ratio": "",
        "contour_extent": "",
        "contour_circularity": "",
        "contour_solidity": "",
        "contour_perimeter_mm": "",
        "bbox_width_mm": "",
        "bbox_height_mm": "",
        "profile_row_index": "",
        "profile_row_sample_count": 0,
        "profile_row_mean_value": "",
        "profile_row_peak_to_valley_value": "",
        "profile_row_Ra_relative": "",
        "profile_row_Rq_relative": "",
        "profile_col_index": "",
        "profile_col_sample_count": 0,
        "profile_col_mean_value": "",
        "profile_col_peak_to_valley_value": "",
        "profile_col_Ra_relative": "",
        "profile_col_Rq_relative": "",
    }

    mask_u8 = contact_mask.astype(np.uint8)
    component_count, _, _, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    features["contact_components"] = int(component_count - 1)
    if component_count <= 1:
        return features

    # 选面积最大的连通接触区域计算轮廓，避免小噪点主导轮廓指标。
    contours, _ = cv2.findContours(
        (mask_u8 * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return features
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

    features["contour_area_px"] = round(contour_area, 2)
    features["contour_perimeter_px"] = round(perimeter, 2)
    features["bbox_width_px"] = int(width)
    features["bbox_height_px"] = int(height)
    features["bbox_aspect_ratio"] = round(
        max(width, height) / float(min(width, height)), 4
    ) if min(width, height) > 0 else ""
    features["contour_extent"] = round(
        contour_area / float(width * height), 4
    ) if width * height > 0 else ""
    features["contour_circularity"] = round(
        min(1.0, 4.0 * np.pi * contour_area / (perimeter ** 2)), 4
    ) if perimeter > 0 else ""
    features["contour_solidity"] = round(
        contour_area / hull_area, 4
    ) if hull_area > 0 else ""

    if mm_per_pixel and mm_per_pixel > 0:
        features["contour_perimeter_mm"] = round(perimeter * mm_per_pixel, 3)
        features["bbox_width_mm"] = round(width * mm_per_pixel, 3)
        features["bbox_height_mm"] = round(height * mm_per_pixel, 3)

    # 选接触像素最多的一行和一列作为代表性横向、纵向剖面。
    row_counts = contact_mask.sum(axis=1)
    col_counts = contact_mask.sum(axis=0)
    row_index = int(np.argmax(row_counts))
    col_index = int(np.argmax(col_counts))

    row_positions = np.flatnonzero(contact_mask[row_index, :])
    row_values = height_map[row_index, row_positions]
    row_stats = profile_statistics(row_values, row_positions)
    features["profile_row_index"] = row_index
    features["profile_row_sample_count"] = row_stats["count"]
    features["profile_row_mean_value"] = row_stats["mean"]
    features["profile_row_peak_to_valley_value"] = row_stats["peak_to_valley"]
    features["profile_row_Ra_relative"] = row_stats["ra"]
    features["profile_row_Rq_relative"] = row_stats["rq"]

    col_positions = np.flatnonzero(contact_mask[:, col_index])
    col_values = height_map[col_positions, col_index]
    col_stats = profile_statistics(col_values, col_positions)
    features["profile_col_index"] = col_index
    features["profile_col_sample_count"] = col_stats["count"]
    features["profile_col_mean_value"] = col_stats["mean"]
    features["profile_col_peak_to_valley_value"] = col_stats["peak_to_valley"]
    features["profile_col_Ra_relative"] = col_stats["ra"]
    features["profile_col_Rq_relative"] = col_stats["rq"]

    return features


def extract_features(height_map, mm_per_pixel=None):
    """从一帧高度矩阵中提取接触区域、中心和高度统计量。"""
    valid = np.isfinite(height_map)
    contact_mask = valid & (height_map > CONTACT_THRESHOLD)
    rows, cols = np.where(contact_mask)
    pixel_count = int(contact_mask.sum())

    features = {
        "contact_pixels": pixel_count,
        "contact_area_mm2": "",
        "center_x_pixel": "",
        "center_y_pixel": "",
        "center_x_mm": "",
        "center_y_mm": "",
        "max_height_value": "",
        "mean_height_value": "",
    }
    # 为轮廓和高度剖面列设置默认值；未检测到接触时这些值保持空白。
    features.update(extract_contour_features(contact_mask, height_map, mm_per_pixel))

    if pixel_count == 0:
        return contact_mask, features

    center_x = float(cols.mean())
    center_y = float(rows.mean())
    selected_height = height_map[contact_mask]
    features["center_x_pixel"] = round(center_x, 2)
    features["center_y_pixel"] = round(center_y, 2)
    features["max_height_value"] = round(float(selected_height.max()), 6)
    features["mean_height_value"] = round(float(selected_height.mean()), 6)

    if mm_per_pixel and mm_per_pixel > 0:
        # 标定文件提供 mm/pixel，因此单像素面积为 (mm/pixel)²。
        features["contact_area_mm2"] = round(
            pixel_count * (mm_per_pixel ** 2), 3
        )
        # 将图像中心作为坐标原点；图像 y 轴向下，因此物理 y 取反。
        features["center_x_mm"] = round(
            (center_x - (height_map.shape[1] - 1) / 2) * mm_per_pixel, 3
        )
        features["center_y_mm"] = round(
            ((height_map.shape[0] - 1) / 2 - center_y) * mm_per_pixel, 3
        )

    return contact_mask, features


def main():
    files = sorted(SHAPE_DATASET_DIR.rglob("*_height.npy"))
    if not files:
        raise FileNotFoundError(
            f"没有找到类别数据：{SHAPE_DATASET_DIR}\n"
            "请先运行 _3_Shape_Reconstruction.py，选择类别并按 s 保存样本。"
        )

    mm_per_pixel = None
    if SCALE_PATH.exists():
        scale_data = np.load(SCALE_PATH)
        # position_scale[2] 在此项目中的实际含义是 mm/pixel，
        # 因为相机代码用它乘以像素坐标来生成毫米坐标。
        mm_per_pixel = float(scale_data[2])
        print(f"相机标定毫米/像素：{mm_per_pixel:.4f}")
    else:
        print("未找到 position_scale.npy；仍计算像素面积和像素中心。")

    output_rows = []
    previous_map = None
    previous_mask = None
    previous_label = None

    print(f"接触阈值：height > {CONTACT_THRESHOLD}")
    print(f"发现 {len(files)} 帧数据：\n")

    for path in files:
        # 类别切换时不计算跨类别的相邻帧差异。
        class_label = path.parent.name
        if class_label != previous_label:
            previous_map = None
            previous_mask = None

        height_map = np.load(path)
        contact_mask, features = extract_features(height_map, mm_per_pixel)
        valid_values = height_map[np.isfinite(height_map)]

        # 逐帧统计量用于检查未按压、按压和松开时数据是否有变化。
        features["class_label"] = class_label
        features["file"] = str(path.relative_to(RESULTS_DIR))
        features["matrix_shape"] = str(height_map.shape)
        features["matrix_min"] = round(float(valid_values.min()), 6)
        features["matrix_max"] = round(float(valid_values.max()), 6)
        features["matrix_mean"] = round(float(valid_values.mean()), 6)
        features["mean_abs_diff_from_previous"] = ""
        features["mask_iou_from_previous"] = ""

        if previous_map is not None and previous_map.shape == height_map.shape:
            valid_pair = np.isfinite(previous_map) & np.isfinite(height_map)
            if valid_pair.any():
                mean_abs_diff = float(
                    np.mean(np.abs(height_map[valid_pair] - previous_map[valid_pair]))
                )
                features["mean_abs_diff_from_previous"] = round(mean_abs_diff, 6)

            union = np.logical_or(contact_mask, previous_mask).sum()
            intersection = np.logical_and(contact_mask, previous_mask).sum()
            iou = 1.0 if union == 0 else float(intersection / union)
            features["mask_iou_from_previous"] = round(iou, 4)

        status = "检测到接触区域" if features["contact_pixels"] else "未检测到接触"
        print(f"[{class_label}] {path.name}  [{status}]")
        print(
            f"  矩阵范围 {features['matrix_min']} ~ {features['matrix_max']}，"
            f"均值 {features['matrix_mean']}，接触像素 {features['contact_pixels']}"
        )
        if features["contact_pixels"]:
            print(
                f"  中心像素 ({features['center_x_pixel']}, "
                f"{features['center_y_pixel']})，"
                f"最大/平均高度值 {features['max_height_value']} / "
                f"{features['mean_height_value']}"
            )
            if features["contact_area_mm2"] != "":
                print(
                    f"  面积约 {features['contact_area_mm2']} mm^2，"
                    f"中心相对坐标 ({features['center_x_mm']}, "
                    f"{features['center_y_mm']}) mm"
                )
            print(
                f"  最大接触轮廓：长宽比 {features['bbox_aspect_ratio']}，"
                f"圆度 {features['contour_circularity']}，"
                f"凸实度 {features['contour_solidity']}"
            )
            print(
                f"  横向剖面 Ra/Rq（相对值）："
                f"{features['profile_row_Ra_relative']} / "
                f"{features['profile_row_Rq_relative']}；"
                f"纵向剖面：{features['profile_col_Ra_relative']} / "
                f"{features['profile_col_Rq_relative']}"
            )
        if features["mean_abs_diff_from_previous"] != "":
            print(
                f"  与上一帧平均绝对差 {features['mean_abs_diff_from_previous']}，"
                f"接触区域 IoU {features['mask_iou_from_previous']}"
            )
        print()

        output_rows.append(features)
        previous_map = height_map
        previous_mask = contact_mask
        previous_label = class_label

    # 写 CSV 方便用 Excel 查看、整理并记录每帧对应的按压状态。
    csv_path = SHAPE_DATASET_DIR / "shape_height_features.csv"
    fieldnames = [
        "class_label", "file", "matrix_shape", "matrix_min", "matrix_max", "matrix_mean",
        "contact_pixels", "contact_area_mm2", "center_x_pixel",
        "center_y_pixel", "center_x_mm", "center_y_mm", "max_height_value",
        "mean_height_value", "contact_components", "contour_area_px",
        "contour_perimeter_px", "bbox_width_px", "bbox_height_px",
        "bbox_aspect_ratio", "contour_extent", "contour_circularity",
        "contour_solidity", "contour_perimeter_mm", "bbox_width_mm",
        "bbox_height_mm", "profile_row_index", "profile_row_sample_count",
        "profile_row_mean_value", "profile_row_peak_to_valley_value",
        "profile_row_Ra_relative", "profile_row_Rq_relative",
        "profile_col_index", "profile_col_sample_count",
        "profile_col_mean_value", "profile_col_peak_to_valley_value",
        "profile_col_Ra_relative", "profile_col_Rq_relative",
        "mean_abs_diff_from_previous",
        "mask_iou_from_previous",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"分析结果已保存到：{csv_path}")
    print(
        "注意：请根据未按压帧的噪声调整 CONTACT_THRESHOLD；"
        "高度值暂不代表牛顿或毫米，需另行确认深度标定单位。"
        "剖面 Ra/Rq 当前是去趋势后的相对起伏值，不是标准粗糙度测量值。"
    )


if __name__ == "__main__":
    main()
