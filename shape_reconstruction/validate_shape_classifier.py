"""使用形状分类模型批量识别验证集，并保存逐样本预测结果。

验证集目录格式：results/shape_validation/<类别>/*_height.npy
例如：results/shape_validation/圆柱/xxx_height.npy

在 PyCharm 中直接运行本脚本，会用文件夹名作为真实标签进行批量评估。
"""

import csv
from pathlib import Path
import sys

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "shape_dataset" / "shape_classifier.joblib"
VALIDATION_DIR = RESULTS_DIR / "shape_validation"
PREDICTIONS_CSV = VALIDATION_DIR / "validation_predictions.csv"
MISCLASSIFIED_CSV = VALIDATION_DIR / "validation_misclassified.csv"

# 确保无论 PyCharm 工作目录如何设置，都从当前脚本导入同一套特征代码。
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from train_shape_classifier import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_PIPELINE_VERSION,
    extract_shape_features,
)


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到形状分类模型：{MODEL_PATH}\n"
            "请先运行 train_shape_classifier.py 训练模型。"
        )

    bundle = joblib.load(MODEL_PATH)
    if (
        bundle.get("feature_names") != FEATURE_NAMES
        or bundle.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION
    ):
        raise ValueError(
            "模型的特征/处理版本与当前代码不匹配。"
            "请重新运行 train_shape_classifier.py 训练模型。"
        )
    return bundle["model"], [str(label) for label in bundle.get("classes", [])]


def find_validation_samples():
    """从每个类别文件夹收集高度图，文件夹名作为真实类别。"""
    samples = []
    if not VALIDATION_DIR.is_dir():
        return samples

    for class_dir in sorted(path for path in VALIDATION_DIR.iterdir() if path.is_dir()):
        for height_path in sorted(class_dir.glob("*_height.npy")):
            samples.append((class_dir.name, height_path))
    return samples


def main():
    try:
        model, model_classes = load_model()
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    samples = find_validation_samples()
    if not samples:
        print(f"验证集目录中没有找到 *_height.npy：{VALIDATION_DIR}")
        print("请先用 _3_Shape_Reconstruction.py 选择验证集并采集样本。")
        return 1

    validation_labels = sorted({label for label, _ in samples})
    unknown_labels = sorted(set(validation_labels) - set(model_classes))
    if unknown_labels:
        print("验证集目录中存在模型未训练过的类别：" + ", ".join(unknown_labels))
        print("模型类别：" + ", ".join(model_classes))
        print("请确认类别文件夹名称，并用这些类别重新训练模型。")
        return 1

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError as error:
        print(f"缺少 scikit-learn：{error}")
        return 1

    rows = []
    skipped = []
    for true_label, height_path in samples:
        try:
            height_map = np.load(height_path, allow_pickle=False)
            features = extract_shape_features(height_map)
        except Exception as error:
            skipped.append((height_path, str(error)))
            continue

        if features is None:
            skipped.append((height_path, "没有检测到有效接触区域"))
            continue
        if features.size != len(FEATURE_NAMES):
            skipped.append((height_path, f"特征维数错误：{features.size}"))
            continue

        predicted_label = str(model.predict(features.reshape(1, -1))[0])
        rows.append({
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": str(true_label == predicted_label),
            "height_path": str(height_path.resolve()),
        })

    if skipped:
        print(f"跳过 {len(skipped)} 个无法识别的样本：")
        for path, reason in skipped:
            print(f"  {path}: {reason}")
    if not rows:
        print("没有可用于评估的验证样本。")
        return 1

    y_true = [row["true_label"] for row in rows]
    y_pred = [row["predicted_label"] for row in rows]
    labels = sorted(set(y_true) | set(y_pred))

    print(f"\n验证集识别结果：{len(rows)} 张有效样本")
    print(f"验证集准确率：{accuracy_score(y_true, y_pred):.3f}")
    print("\n分类报告：")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("混淆矩阵（行=真实类别，列=预测类别）：")
    print(labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))

    misclassified = [row for row in rows if row["correct"] == "False"]
    fields = ["true_label", "predicted_label", "correct", "height_path"]
    for output_path, output_rows in (
        (PREDICTIONS_CSV, rows),
        (MISCLASSIFIED_CSV, misclassified),
    ):
        with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    print(f"\n逐样本结果：{PREDICTIONS_CSV}")
    print(f"误判样本清单：{MISCLASSIFIED_CSV}")
    if misclassified:
        print("误判样本：")
        for row in misclassified:
            print(
                f"  真实：{row['true_label']}，预测：{row['predicted_label']}，"
                f"文件：{row['height_path']}"
            )
    else:
        print("本次验证集没有误判样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
