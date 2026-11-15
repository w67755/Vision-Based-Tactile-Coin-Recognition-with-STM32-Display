"""Train a tactile shape classifier from labeled shape_dataset folders.

Run this file in the project's Python interpreter after:
1) collecting samples in results/shape_dataset/<class_name>/;
2) running analyze_height_maps.py to create shape_height_features.csv.

The script uses each folder name as its class label and reloads each .npy
height map to compute shape descriptors. It reports stratified cross-validation
results, then trains on all usable samples and saves the final pipeline as
results/shape_dataset/shape_classifier.joblib.
"""

import csv
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATASET_DIR = RESULTS_DIR / "shape_dataset"
CSV_PATH = DATASET_DIR / "shape_height_features.csv"
MODEL_PATH = DATASET_DIR / "shape_classifier.joblib"
CV_PREDICTIONS_PATH = DATASET_DIR / "shape_cv_predictions.csv"
CV_MISCLASSIFIED_PATH = DATASET_DIR / "shape_cv_misclassified.csv"
CONTACT_THRESHOLD = 0.08
RANDOM_STATE = 42
RELIEF_GRID_SIZE = 4
MEDIAN_KERNEL_SIZE = 3
MIN_COMPONENT_AREA = 20
FEATURE_PIPELINE_VERSION = "median3_component20_relief4x4_v1"

CONTOUR_FEATURE_NAMES = [
    "log_contact_pixels",
    "aspect_ratio",
    "extent",
    "circularity",
    "solidity",
    "relative_height_spread",
]
# 将接触区域裁剪并归一化到固定网格后，记录每格的平均浮雕高度和局部起伏。
# 总特征数为 6 个轮廓特征 + 16 个局部高度均值 + 16 个局部起伏值。
RELIEF_FEATURE_NAMES = (
    [f"relief_mean_{row}_{col}" for row in range(RELIEF_GRID_SIZE)
     for col in range(RELIEF_GRID_SIZE)]
    + [f"relief_std_{row}_{col}" for row in range(RELIEF_GRID_SIZE)
       for col in range(RELIEF_GRID_SIZE)]
)
FEATURE_NAMES = CONTOUR_FEATURE_NAMES + RELIEF_FEATURE_NAMES


def extract_shape_features(height_map):
    """Combine contact silhouette features with normalized local relief features."""
    # 先对高度图做 3×3 中值滤波，压制孤立像素噪声；原始 .npy 不会被修改。
    source_map = np.asarray(height_map, dtype=np.float32)
    valid = np.isfinite(source_map)
    filtered_map = cv2.medianBlur(
        np.nan_to_num(source_map, nan=0.0, posinf=0.0, neginf=0.0),
        MEDIAN_KERNEL_SIZE,
    )
    mask = (valid & (filtered_map > CONTACT_THRESHOLD)).astype(np.uint8)

    # 删除小于 20 像素的孤立连通区域，避免零散噪点影响轮廓统计。
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    clean_mask = np.zeros_like(mask)
    for component_id in range(1, component_count):
        if component_stats[component_id, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean_mask[component_labels == component_id] = 1
    mask = clean_mask

    contact_pixels = int(mask.sum())
    if contact_pixels == 0:
        return None

    contours_result = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_result[-2]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if width == 0 or height == 0:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    circularity = (
        4.0 * np.pi * contact_pixels / (perimeter ** 2)
        if perimeter > 0 else 0.0
    )
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = min(1.0, contact_pixels / hull_area) if hull_area > 0 else 0.0
    extent = contact_pixels / float(width * height)
    aspect_ratio = max(width, height) / float(min(width, height))

    heights = filtered_map[mask.astype(bool)]
    p05, p95 = np.percentile(heights, [5, 95])
    height_span = float(p95 - p05)
    relative_height_spread = (
        float(np.std(heights)) / height_span if height_span > 1e-8 else 0.0
    )

    contour_features = np.asarray([
        np.log1p(contact_pixels),
        aspect_ratio,
        extent,
        circularity,
        solidity,
        relative_height_spread,
    ], dtype=np.float64)

    # 以最大接触轮廓的外接框为 ROI，减弱物体在传感器画面中的绝对位置影响。
    # 将高度按接触区域的 5%～95% 分位数归一化，降低整体按压深浅的影响。
    roi_height = filtered_map[y:y + height, x:x + width].astype(np.float64)
    roi_mask = mask[y:y + height, x:x + width].astype(bool)
    roi_values = roi_height[roi_mask]
    relief_low, relief_high = np.percentile(roi_values, [5, 95])
    relief_span = float(relief_high - relief_low)
    if relief_span > 1e-8:
        normalized_roi = np.clip((roi_height - relief_low) / relief_span, 0.0, 1.0)
    else:
        normalized_roi = np.zeros_like(roi_height, dtype=np.float64)

    # 统计 4×4 网格内的归一化高度均值和标准差，保留局部浮雕分布，
    # 同时避免把整张高分辨率图直接输入模型造成特征过多。
    relief_means = []
    relief_stds = []
    for grid_row in range(RELIEF_GRID_SIZE):
        row_start = grid_row * height // RELIEF_GRID_SIZE
        row_end = (grid_row + 1) * height // RELIEF_GRID_SIZE
        for grid_col in range(RELIEF_GRID_SIZE):
            col_start = grid_col * width // RELIEF_GRID_SIZE
            col_end = (grid_col + 1) * width // RELIEF_GRID_SIZE
            cell_mask = roi_mask[row_start:row_end, col_start:col_end]
            cell_values = normalized_roi[row_start:row_end, col_start:col_end][cell_mask]
            if cell_values.size:
                relief_means.append(float(np.mean(cell_values)))
                relief_stds.append(float(np.std(cell_values)))
            else:
                # 圆形接触区外的空网格用 0 填充；轮廓特征同时描述区域形状。
                relief_means.append(0.0)
                relief_stds.append(0.0)

    return np.concatenate([
        contour_features,
        np.asarray(relief_means, dtype=np.float64),
        np.asarray(relief_stds, dtype=np.float64),
    ])


def main():
    if not CSV_PATH.exists():
        print(f"找不到特征表：{CSV_PATH}")
        print("请先运行 analyze_height_maps.py 生成带标签的 CSV。")
        return 1

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not {"class_label", "file"}.issubset(reader.fieldnames):
            print("CSV 缺少 class_label 或 file 列，请重新运行 analyze_height_maps.py。")
            return 1
        table = [row for row in reader if row.get("class_label") and row.get("file")]

    if not table:
        print("特征表中没有带类别标签的样本，请先运行 analyze_height_maps.py。")
        return 1

    if not {"class_label", "file"}.issubset(table[0]):
        print("CSV 缺少 class_label 或 file 列，请重新运行 analyze_height_maps.py。")
        return 1

    counts = Counter(row["class_label"] for row in table)
    print("CSV 中各类别样本数：")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    if len(counts) < 2:
        print("\n目前只有一个类别，无法训练分类器。请至少再采集一种不同形状，")
        print("将数据存入对应类别文件夹，并重新运行 analyze_height_maps.py。")
        return 0
    if any(count < 4 for count in counts.values()):
        small = ", ".join(
            f"{label}({count})" for label, count in sorted(counts.items()) if count < 4
        )
        print(f"\n以下类别样本少于 4 组：{small}")
        print("请先补充样本，再训练并划分训练集/测试集。")
        return 0

    # scikit-learn is only needed once there are at least two usable classes.
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        import joblib
    except ImportError as error:
        print(f"缺少分类依赖：{error}")
        print("请在项目使用的 Python 解释器中安装 scikit-learn。")
        return 1

    X_rows = []
    y_rows = []
    sample_paths = []
    skipped = []
    for row in table:
        relative_path = Path(row["file"])
        height_path = RESULTS_DIR / relative_path
        if not height_path.exists():
            skipped.append((str(relative_path), "文件不存在"))
            continue
        try:
            height_map = np.load(height_path)
            features = extract_shape_features(height_map)
        except Exception as error:
            skipped.append((str(relative_path), str(error)))
            continue
        if features is None:
            skipped.append((str(relative_path), "未检测到有效接触区域"))
            continue
        X_rows.append(features)
        y_rows.append(str(row["class_label"]))
        sample_paths.append(height_path.resolve())

    if skipped:
        print(f"\n跳过 {len(skipped)} 个无法使用的样本。")
        for path, reason in skipped[:10]:
            print(f"  {path}: {reason}")
        if len(skipped) > 10:
            print("  其余问题略。")

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows)
    usable_labels, usable_counts = np.unique(y, return_counts=True)
    if len(usable_labels) < 2 or (usable_counts < 4).any():
        print("\n有效高度图不足以支持分层训练/测试划分，请补齐各类别有效样本。")
        return 0

    # 分层 K 折交叉验证：每一折都尽量按相同类别比例划分。
    # 某类样本数不足 5 时，折数自动降低到该类样本数。
    n_splits = min(5, int(usable_counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
    )
    # 每个样本只在它没有参与训练的那一折中被预测一次。
    predictions = cross_val_predict(cv_model, X, y, cv=cv)

    labels = sorted(np.unique(y).tolist())
    print(f"\n分层交叉验证：{n_splits} 折；总样本：{len(y)}")
    print(f"交叉验证准确率：{accuracy_score(y, predictions):.3f}")
    print("\n交叉验证分类报告：")
    print(classification_report(y, predictions, labels=labels, zero_division=0))
    print("交叉验证混淆矩阵（行=真实类别，列=预测类别；顺序见下方类别列表）：")
    print(labels)
    print(confusion_matrix(y, predictions, labels=labels))

    # 保存每个样本的折外预测，便于定位误判样本并查看对应原图。
    prediction_rows = []
    misclassified_rows = []
    for height_path, true_label, predicted_label in zip(sample_paths, y, predictions):
        height_path = Path(height_path)
        row = {
            "true_label": str(true_label),
            "predicted_label": str(predicted_label),
            "correct": str(true_label == predicted_label),
            "height_path": str(height_path),
            "gray_path": str(height_path.with_name(height_path.name.replace("_height.npy", "_gray.png"))),
            "depth_path": str(height_path.with_name(height_path.name.replace("_height.npy", "_depth.png"))),
        }
        prediction_rows.append(row)
        if true_label != predicted_label:
            misclassified_rows.append(row)

    prediction_fields = [
        "true_label", "predicted_label", "correct",
        "height_path", "gray_path", "depth_path",
    ]
    for output_path, rows in (
        (CV_PREDICTIONS_PATH, prediction_rows),
        (CV_MISCLASSIFIED_PATH, misclassified_rows),
    ):
        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=prediction_fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n逐样本交叉验证结果：{CV_PREDICTIONS_PATH}")
    print(f"误判样本清单：{CV_MISCLASSIFIED_PATH}")
    if misclassified_rows:
        print("误判样本（真实类别 -> 预测类别）：")
        for row in misclassified_rows:
            print(f"  {row['true_label']} -> {row['predicted_label']}: {row['height_path']}")
            print(f"    灰度图：{row['gray_path']}")
            print(f"    深度图：{row['depth_path']}")
    else:
        print("本次交叉验证没有误判样本。")

    # 交叉验证只用于评估。保存的最终模型使用全部有效样本重新训练，
    # 供之后对新采集的高度图进行预测。
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "contact_threshold": CONTACT_THRESHOLD,
        "median_kernel_size": MEDIAN_KERNEL_SIZE,
        "min_component_area": MIN_COMPONENT_AREA,
        "classes": labels,
    }, MODEL_PATH)
    print(f"\n模型已保存到：{MODEL_PATH}")
    print("注意：交叉验证只反映当前数据及采集条件；若同一物体或相邻按压样本相似，结果可能偏乐观。")
    print("保存的模型用全部有效样本训练，不能再用这批样本评估模型效果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
