"""实时识别球、圆柱和正方体；稳定结果显示在电脑并可发送给 STM32。"""

import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import serial
import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for import_path in (PROJECT_ROOT, BASE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shape_reconstruction import Sensor  # noqa: E402
from train_shape_classifier import (  # noqa: E402
    CONTACT_THRESHOLD,
    FEATURE_NAMES,
    FEATURE_PIPELINE_VERSION,
    extract_shape_features,
)

MODEL_PATH = BASE_DIR / "results" / "shape_dataset" / "shape_classifier.joblib"
CONFIG_PATH = BASE_DIR / "shape_config.yaml"

# 【串口配置】填写设备管理器中 USB 转 TTL 模块对应的 COM 端口。
SERIAL_PORT = "COM7"
SERIAL_BAUDRATE = 9600

MIN_CONTACT_PIXELS = 1000
STABILIZE_SECONDS = 2.0
VOTE_WINDOW = 15
VOTE_REQUIRED = 10
RELEASE_FRAMES_TO_RESET = 5

DISPLAY_NAMES = {
    "球": "SPHERE",
    "圆柱": "CYLINDER",
    "正方体": "CUBE",
}

# 协议格式为 ASCII 文本，以换行符结尾；需与 STM32 main.c 保持一致。
SHAPE_PACKETS = {
    "球": b"SHAPE,SPHERE\n",
    "圆柱": b"SHAPE,CYLINDER\n",
    "正方体": b"SHAPE,CUBE\n",
}


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到形状分类模型：{MODEL_PATH}\n"
            "请先运行 train_shape_classifier.py 训练当前形状类别。"
        )

    bundle = joblib.load(MODEL_PATH)
    if (
        bundle.get("feature_names") != FEATURE_NAMES
        or bundle.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION
    ):
        raise ValueError("模型特征版本与当前代码不匹配，请重新运行 train_shape_classifier.py。")

    model = bundle["model"]
    model_labels = {str(label) for label in getattr(model, "classes_", [])}
    expected_labels = set(SHAPE_PACKETS)
    if model_labels != expected_labels:
        raise ValueError(
            "模型类别不是当前的球、圆柱、正方体。"
            f"模型类别：{sorted(model_labels)}；请重新训练后再运行实时识别。"
        )
    return model


def main():
    try:
        model = load_model()
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    if not CONFIG_PATH.is_file():
        print(f"找不到相机配置：{CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)
    cfg["calibration_root_dir"] = str(
        (BASE_DIR / cfg["calibration_root_dir"]).resolve()
    ).replace("\\", "/")

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        print(f"串口已打开：{SERIAL_PORT}，{SERIAL_BAUDRATE} baud")
    except serial.SerialException as error:
        print(f"串口打开失败（仍可在电脑上识别）：{error}")

    sensor = None
    try:
        print("正在初始化相机和传感器，请保持传感器空载且不要触碰表面。")
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
            f"形状实时识别已启动：放上物体并保持不动 {STABILIZE_SECONDS:.0f} 秒，"
            f"再用 {VOTE_WINDOW} 帧投票；松开后可识别下一个物体；按 q 退出。"
        )

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
                    status = f"Hold still: {STABILIZE_SECONDS - elapsed:.1f}s"
                else:
                    features = extract_shape_features(height_map)
                    if features is not None and features.size == len(FEATURE_NAMES):
                        label = str(model.predict(features.reshape(1, -1))[0])
                        predictions.append(label)

                    if len(predictions) == VOTE_WINDOW and not announced:
                        most_common_label, vote_count = Counter(predictions).most_common(1)[0]
                        final_label = most_common_label
                        final_vote_count = vote_count
                        low_consensus = vote_count < VOTE_REQUIRED
                        if low_consensus:
                            print(
                                f"识别不稳定：倾向{final_label}，"
                                f"仅 {vote_count}/{VOTE_WINDOW} 帧一致；请松开后重新放置。"
                            )
                        else:
                            print(
                                f"识别结果：{final_label} "
                                f"（{vote_count}/{VOTE_WINDOW} 帧一致）"
                            )
                            packet = SHAPE_PACKETS[final_label]
                            if ser is not None and ser.is_open:
                                try:
                                    ser.write(packet)
                                    ser.flush()
                                    print(f"已发送给 STM32：{packet.decode('ascii').strip()}")
                                except serial.SerialException as error:
                                    print(f"串口发送失败：{error}")
                            else:
                                print("串口未连接，结果只显示在电脑上。")
                        announced = True

                    if final_label is not None:
                        if low_consensus:
                            status = f"LOW CONSENSUS: {DISPLAY_NAMES[final_label]} ({final_vote_count}/{VOTE_WINDOW})"
                        else:
                            status = f"RESULT: {DISPLAY_NAMES[final_label]}"
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
                    status = "Release / place shape"
                else:
                    status = "Keep shape in place..."

            cv2.putText(
                display, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                display, "Press Q to quit", (12, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow("Shape Recognition", display)
            cv2.imshow("RawImage_GRAY", gray)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except serial.SerialException as error:
        print(f"相机/串口运行错误：{error}")
        return 1
    finally:
        if sensor is not None and sensor.cap is not None:
            sensor.cap.release()
        if ser is not None and ser.is_open:
            ser.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
