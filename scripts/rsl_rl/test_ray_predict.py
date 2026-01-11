# -*- coding: utf-8 -*-
"""
单样本测试脚本：
1) 读取你指定的 .npy 深度图，并从同目录（或指定路径）的 label.pkl 读取真实 11 维射线距离
2) 加载你训练好的 TorchScript 模型，输出预测的 11 维射线距离
3) 画柱状图对比：预测(蓝色) vs 真实(红色)
"""

import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    # ===================== 1) 读取的 npy 文件路径 =====================
    npy_path = "/home/qihang/code_lab/unitree_rl_lab-main/logs/rsl_rl/collection_data/rec_cam_2026-01-10_19-07-45/robot_0_step245.npy"
    # =====================================================================================

    # （可选）如果你的 label.pkl 不在 npy 同目录，在这里手动指定；否则保持 None 自动使用同目录下的 label.pkl
    pkl_path = None  # 例如："/home/qihang/.../rec_cam_xxx/label.pkl"

    # ===================== 2) 加载的网络模型路径 =====================
    model_path = "/home/qihang/code_lab/unitree_rl_lab-main/logs/rsl_rl/ray_prediction_final/20260111-115935_resnet18_N20007/epoch200.pt"
    # =====================================================================================

    # 输出图片保存目录（可改）
    out_dir = "ray_pred_test_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 自动定位 label.pkl ----------
    if pkl_path is None:
        pkl_path = os.path.join(os.path.dirname(os.path.abspath(npy_path)), "label.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"找不到 label.pkl: {pkl_path}")

    # ---------- 读取真实标签 ----------
    with open(pkl_path, "rb") as f:
        labels = pickle.load(f)

    key = os.path.splitext(os.path.basename(npy_path))[0]
    if key not in labels:
        raise KeyError(f"label.pkl 中找不到 key='{key}'。请确认 npy 文件名与 label.pkl 的键一致。")

    y_true = np.asarray(labels[key], dtype=np.float32).reshape(-1)
    if y_true.shape[0] != 11:
        print(f"[警告] 真实标签维度是 {y_true.shape[0]}，不是 11。仍将继续画图。")

    # ---------- 读取深度图 ----------
    depth = np.load(npy_path).astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]  # (H, W)

    # ---------- 构造网络输入 (1,1,H,W) ----------
    x = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)

    # ===================== 加载网络模型（TorchScript）并做推理（中文标注） =====================
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    with torch.no_grad():
        y_pred = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    # =====================================================================================

    if y_pred.shape[0] != y_true.shape[0]:
        print(f"[警告] 预测维度 {y_pred.shape[0]} 与真实维度 {y_true.shape[0]} 不一致，将按较短维度对齐画图。")
        n = min(y_pred.shape[0], y_true.shape[0])
        y_pred = y_pred[:n]
        y_true = y_true[:n]
    else:
        n = y_true.shape[0]  # ← 关键补这一行

    # 误差
    error = np.abs(y_pred - y_true)

    # ===================== 3) 1.画柱状图：预测(蓝) vs 真实(红) =====================
    idx = np.arange(n)
    bar_w = 0.4

    plt.figure(figsize=(10, 4))
    plt.bar(idx - bar_w / 2, y_true, width=bar_w, color="red", label="ray_real")
    plt.bar(idx + bar_w / 2, y_pred, width=bar_w, color="blue", label="ray_predict")
    plt.xticks(idx, [str(i) for i in idx])
    plt.xlabel("Ray index")
    plt.ylabel("Ray distance")
    plt.title(f"Ray Prediction vs Ground Truth\nkey={key}")
    plt.legend()

    plt.tight_layout()
    plt.show()
    # ======================================================================

    # ===================== 图2：误差表格（1 行 11 列） =====================
    plt.figure(figsize=(12, 2))
    plt.axis("off")

    table = plt.table(
        cellText=[["{:.4f}".format(e) for e in error]],
        colLabels=[f"ray{i}" for i in range(n)],
        rowLabels=["error"],
        loc="center",
        cellLoc="center",
    )

    table.scale(1, 2)
    # plt.title(f"Absolute Error per Ray\nkey={key}", pad=20)
    plt.title(f"Absolute Error per Ray (unit: m)\nkey={key}", pad=20)
    plt.show()
    # ======================================================================


    print("[OK] npy:", os.path.abspath(npy_path))
    print("[OK] pkl:", os.path.abspath(pkl_path))
    print("[OK] model:", os.path.abspath(model_path))
    print("[OK] y_true:", y_true)
    print("[OK] y_pred:", y_pred)
    print("[OK] error:", error)
    print("[SAVED]", fig1_path)
    print("[SAVED]", fig2_path)



if __name__ == "__main__":
    main()
