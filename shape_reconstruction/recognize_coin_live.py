"""硬币实时触觉识别：稳定接触后用多数投票识别，并在电脑窗口显示结果。

请将本文件放到项目的 shape_reconstruction 目录，先运行
train_shape_classifier.py 生成匹配的 shape_classifier.joblib。
启动时先让传感器保持空载，在参考图窗口按 y；随后放入硬币，按 q 退出。
本脚本只识别，不保存采集数据。
"""

from collections import Counter, deque
from pathlib import Path
import sys
import time

import cv2
import joblib
import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parent
# 同时加入项目根目录和本目录，兼容 PyCharm 工作目录设置。
sys.path.insert(0, str(BASE_DIR.parent))
sys.path.insert(0, str(BASE_DIR))

from shape_reconstruction import Sensor
from train_shape_classifier import (
    FEATURE_NAMES,
    FEATURE_PIPELINE_VERSION,
    CONTACT_THRESHOLD,
    extract_shape_features,
)

MODEL_PATH = BASE_DIR / "results" / "shape_dataset" / "shape_classifier.joblib"
CONFIG_PATH = BASE_DIR / "shape_config.yaml"

# 接触区域最少像素数；若空载时误认为有接触，可适当调大。
MIN_CONTACT_PIXELS = 1000
STABILIZE_SECONDS = 2.0
VOTE_WINDOW = 15
VOTE_REQUIRED = 10
RELEASE_FRAMES_TO_RESET = 5

DISPLAY_NAMES = {
    "1元": "1 YUAN",
    "5角": "5 JIAO",
    "1角": "1 JIAO",
}


def main():
    if not MODEL_PATH.is_file():
        print(f"找不到模型：{MODEL_PATH}")
        print("请先运行 train_shape_classifier.py 训练模型。")
        return 1
    if not CONFIG_PATH.is_file():
        print(f"找不到相机配置：{CONFIG_PATH}")
        return 1

    bundle = joblib.load(MODEL_PATH)
    if (
        bundle.get("feature_names") != FEATURE_NAMES
        or bundle.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION
    ):
        print("模型的特征或去噪版本不匹配，请重新运行 train_shape_classifier.py。")
        return 1
    model = bundle["model"]

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    # 让标定目录始终相对于本配置文件解析，避免 PyCharm 工作目录影响。
    cfg["calibration_root_dir"] = str(
        (BASE_DIR / cfg["calibration_root_dir"]).resolve()
    ).replace("\\", "/")

    print("正在初始化相机。请先保持传感器空载，在参考图窗口按 y。")
    sensor = Sensor(cfg)
    if not sensor.cap.isOpened():
        print("相机未打开，请检查相机连接和 camera_channel 设置。")
        return 1

    predictions = deque(maxlen=VOTE_WINDOW)
    final_label = None
    final_vote_count = 0
    low_consensus = False
    announced = False
    release_frames = 0
    contact_started_at = None

    print(
        f"实时识别已启动：放上硬币后保持不动 {STABILIZE_SECONDS:.0f} 秒，"
        f"程序再采集 {VOTE_WINDOW} 帧投票；松开后识别下一枚；按 q 退出。"
    )
    try:
        while sensor.cap.isOpened():
            image = sensor.get_rectify_crop_image()
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            raw_height_map = sensor.raw_image_2_height_map(gray)
            depth_map = sensor.height_map_2_depth_map(raw_height_map)
            height_map = sensor.expand_image(raw_height_map)

            contact_mask = np.isfinite(height_map) & (height_map > CONTACT_THRESHOLD)
            contact_pixels = int(np.count_nonzero(contact_mask))
            display = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

            if contact_pixels >= MIN_CONTACT_PIXELS:
                release_frames = 0
                if contact_started_at is None:
                    contact_started_at = time.monotonic()
                    predictions.clear()
                    final_label = None
                    final_vote_count = 0
                    low_consensus = False
                    announced = False

                elapsed = time.monotonic() - contact_started_at
                if elapsed < STABILIZE_SECONDS:
                    remaining = STABILIZE_SECONDS - elapsed
                    status = f"Hold still: {remaining:.1f}s"
                else:
                    features = extract_shape_features(height_map)
                    if features is not None and features.size == len(FEATURE_NAMES):
                        label = str(model.predict(features.reshape(1, -1))[0])
                        predictions.append(label)

                    if len(predictions) == VOTE_WINDOW:
                        most_common_label, vote_count = Counter(predictions).most_common(1)[0]
                        if not announced:
                            final_label = most_common_label
                            final_vote_count = vote_count
                            low_consensus = vote_count < VOTE_REQUIRED
                            if low_consensus:
                                print(
                                    f"结果不稳定：倾向{final_label}，"
                                    f"但只有 {vote_count}/{VOTE_WINDOW} 帧一致；建议重新放置再试。"
                                )
                            else:
                                print(
                                    f"识别结果：{final_label}"
                                    f"（等待稳定后 {vote_count}/{VOTE_WINDOW} 帧一致）"
                                )
                            announced = True

                    if final_label:
                        if low_consensus:
                            status = (
                                f"LOW CONSENSUS: {DISPLAY_NAMES.get(final_label, final_label)} "
                                f"({final_vote_count}/{VOTE_WINDOW})"
                            )
                        else:
                            status = f"RESULT: {DISPLAY_NAMES.get(final_label, final_label)}"
                    else:
                        status = f"Collecting votes: {len(predictions)}/{VOTE_WINDOW}"
            else:
                release_frames += 1
                if release_frames >= RELEASE_FRAMES_TO_RESET:
                    predictions.clear()
                    final_label = None
                    final_vote_count = 0
                    low_consensus = False
                    announced = False
                    contact_started_at = None
                    status = "Release / place coin"
                else:
                    status = "Keep coin in place..."

            cv2.putText(
                display, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                display, "Press Q to quit", (12, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow("Coin Recognition", display)
            cv2.imshow("RawImage_GRAY", gray)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        sensor.cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
