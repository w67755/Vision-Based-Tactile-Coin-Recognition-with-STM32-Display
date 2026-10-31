"""逐张浏览验证集高度矩阵，并用统一颜色范围比较样本。 

按 n 看下一张，按 p 看上一张，按 q 或 Esc 退出。不会修改原始 .npy。
"""
from pathlib import Path
import cv2
import numpy as np

# 递归查找验证集各面额目录中的高度矩阵。
data_dir = Path(__file__).resolve().parent / "results" / "shape_validation"
height_files = sorted(data_dir.rglob("*.npy"), key=lambda path: path.stat().st_mtime)
if not height_files:
    raise FileNotFoundError(
        f"没有在验证集目录找到 .npy 文件：{data_dir}\n"
        "请确认数据已保存到 results/shape_validation/ 面额子文件夹中。"
    )

# 先扫描所有样本，求整个验证集共享的最小/最大高度。
# 之后每张图都用这同一范围映射颜色，避免逐图自动拉伸造成误比。
shared_min = float("inf")
shared_max = float("-inf")
valid_file_count = 0
for path in height_files:
    sample = np.load(path, allow_pickle=False)
    if sample.ndim != 2 or sample.size == 0:
        print(f"跳过无效或非二维矩阵：{path}，shape={sample.shape}")
        continue
    finite_values = sample[np.isfinite(sample)]
    if finite_values.size == 0:
        print(f"跳过没有有效数值的矩阵：{path}")
        continue
    shared_min = min(shared_min, float(finite_values.min()))
    shared_max = max(shared_max, float(finite_values.max()))
    valid_file_count += 1

if valid_file_count == 0:
    raise ValueError(f"目录中没有可显示的二维高度矩阵：{data_dir}")

print(f"验证集统一颜色范围：{shared_min:.6f} 到 {shared_max:.6f}")
print(f"找到 {len(height_files)} 个 .npy 文件，其中 {valid_file_count} 个可显示。")

window_name = "Height map viewer - n: next, p: previous, q: quit"
index = 0

def show_height_map(file_index: int) -> None:
    """显示指定高度图，并打印文件名与矩阵统计信息。"""
    height_path = height_files[file_index]
    height_map = np.load(height_path, allow_pickle=False)
    if height_map.size == 0 or height_map.ndim != 2:
        print(f"当前文件不是有效二维矩阵：{height_path}，shape={height_map.shape}")
        return

    print("\n" + "-" * 60)
    print(f"文件 {file_index + 1}/{len(height_files)}：{height_path}")
    print(f"矩阵尺寸：{height_map.shape}")
    print(f"最小值：{np.nanmin(height_map):.4f}")
    print(f"最大值：{np.nanmax(height_map):.4f}")
    print(f"平均值：{np.nanmean(height_map):.4f}")

    # 使用整个验证集共享的范围映射到 0~255，不对单张图单独拉伸。
    display_data = np.nan_to_num(
        height_map.astype(np.float32), nan=shared_min,
        posinf=shared_max, neginf=shared_min,
    )
    if shared_max > shared_min:
        scaled = (display_data - shared_min) * (255.0 / (shared_max - shared_min))
    else:
        scaled = np.zeros_like(display_data, dtype=np.float32)
    gray = np.clip(scaled, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.putText(
        heatmap,
        f"{file_index + 1}/{len(height_files)}  n:next p:prev q:quit",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.imshow(window_name, heatmap)

show_height_map(index)
while True:
    key = cv2.waitKey(0) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        break
    if key in (ord("n"), ord("N")):
        index = (index + 1) % len(height_files)
        show_height_map(index)
    elif key in (ord("p"), ord("P")):
        index = (index - 1) % len(height_files)
        show_height_map(index)
cv2.destroyAllWindows()
