"""使用已训练的触觉分类模型预测一张新采集的高度图面额。

使用前先运行 train_shape_classifier.py 生成模型，然后运行本文件，
在控制台输入一张未加入训练集的 *_height.npy 文件完整路径。
"""

from pathlib import Path
import sys

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
# PyCharm 的工作目录可能与脚本目录不同，因此显式添加本脚本所在目录。
sys.path.insert(0, str(BASE_DIR))

from train_shape_classifier import (
    FEATURE_NAMES,
    FEATURE_PIPELINE_VERSION,
    extract_shape_features,
)


MODEL_PATH = BASE_DIR / "results" / "shape_dataset" / "shape_classifier.joblib"


def main():
    if not MODEL_PATH.exists():
        print(f"找不到模型：{MODEL_PATH}")
        print("请先运行 train_shape_classifier.py 训练并保存模型。")
        return 1

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    saved_feature_names = bundle.get("feature_names", [])

    # 预测端必须与训练端使用同一套特征定义，防止新旧模型维数不一致。
    if (
        saved_feature_names != FEATURE_NAMES
        or bundle.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION
    ):
        print("模型的特征或去噪版本与当前代码不一致。")
        print("请重新运行 train_shape_classifier.py 生成匹配的新模型。")
        return 1

    sample_text = input("请输入一张新采集的 *_height.npy 完整路径：").strip().strip('"')
    height_path = Path(sample_text)
    if not height_path.is_file():
        print(f"找不到高度图文件：{height_path}")
        return 1

    try:
        height_map = np.load(height_path)
        features = extract_shape_features(height_map)
    except Exception as error:
        print(f"读取或处理高度图失败：{error}")
        return 1

    if features is None:
        print("这张图没有检测到有效接触区域，请检查按压是否成功。")
        return 1

    if features.size != len(saved_feature_names):
        print(
            f"特征数量不匹配：当前提取到 {features.size} 项，"
            f"模型需要 {len(saved_feature_names)} 项。"
        )
        return 1

    prediction = str(model.predict(features.reshape(1, -1))[0])
    print(f"预测面额：{prediction}")
    print("提示：这是单次预测；建议同一枚硬币重新放置并预测多次，观察结果是否稳定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
