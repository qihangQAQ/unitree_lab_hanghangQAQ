import os
import time
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm


class CamrecFolderDataset(Dataset):
    """
    读取单个 rec_cam_xxx 文件夹：
      - *.npy: depth image
      - label.pkl: dict[key -> ray_vec]
    每个样本返回：
      x: (1,H,W) float32
      y: (D,) float32
    """
    def __init__(self, rec_dir: str, use_log2: bool = False, log_eps: float = 1e-6):
        self.rec_dir = rec_dir
        self.use_log2 = use_log2
        self.log_eps = log_eps

        label_path = os.path.join(rec_dir, "label.pkl")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"label.pkl not found: {label_path}")

        with open(label_path, "rb") as f:
            self.labels: Dict[str, np.ndarray] = pickle.load(f)

        # collect npy files with matching label keys
        npy_files = [p for p in os.listdir(rec_dir) if p.endswith(".npy")]
        npy_files.sort()

        samples: List[Tuple[str, np.ndarray]] = []
        out_dim = None
        for fn in npy_files:
            key = os.path.splitext(fn)[0]
            if key not in self.labels:
                continue
            y = np.asarray(self.labels[key], dtype=np.float32).reshape(-1)
            if out_dim is None:
                out_dim = int(y.shape[0])
            elif int(y.shape[0]) != out_dim:
                raise ValueError(f"Label dim mismatch: {key} has {y.shape[0]} vs expected {out_dim}")
            samples.append((os.path.join(rec_dir, fn), y))

        if not samples:
            raise RuntimeError(f"No matched samples found in {rec_dir}. Check npy names and label.pkl keys.")

        self.samples = samples
        self.out_dim = out_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        depth_path, y = self.samples[idx]

        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]  # (H,W)

        if self.use_log2:
            # 防止 log2(0) / log2(负数)
            depth = np.log2(np.maximum(depth, 0.0) + self.log_eps)
            y = np.log2(np.maximum(y, 0.0) + self.log_eps)

        x = torch.from_numpy(depth).unsqueeze(0)  # (1,H,W)
        y = torch.from_numpy(y)                   # (D,)
        return x, y

# 网络定义
class Depth2RayResNet(nn.Module):
    def __init__(self, out_dim: int, arch: str = "resnet18"):
        super().__init__()
        if arch == "resnet18":
            # 加载预训练的ResNet18模型
            net = models.resnet18(weights="IMAGENET1K_V1")
        elif arch == "resnet34":
            # 加载预训练的ResNet34模型
            net = models.resnet34(weights="IMAGENET1K_V1")
        else:
            # 不支持的架构报错
            raise ValueError(f"Unsupported arch: {arch}")

        # 获取原模型全连接层的输入维度
        in_features = net.fc.in_features
        # 替换全连接层，适配输出维度
        net.fc = nn.Linear(in_features, out_dim)
        # 将修改后的网络保存为实例属性
        self.net = net
    # 前向传播
    def forward(self, x):
        # x: (B,1,H,W) -> (B,3,H,W)
        x = x.repeat(1, 3, 1, 1)
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser()

    # 你的固定需求（可改默认，也可命令行覆盖）
    p.add_argument("--data_dir", type=str, required=True,
                   help="rec_cam_xxx folder path containing *.npy and label.pkl")
    p.add_argument("--save_root", type=str, default="logs/rsl_rl/ray_prediction",
                   help="root folder for all ray prediction experiments")

    # 训练超参
    p.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # 预处理选项
    p.add_argument("--use_log2", action="store_true",
                   help="apply log2(x+eps) to both depth and labels (default off)")
    p.add_argument("--log_eps", type=float, default=1e-6)

    # 保存与日志
    p.add_argument("--save_every", type=int, default=10, help="save model every N epochs")
    p.add_argument("--exp_name", type=str, default="", help="optional experiment name suffix")
    return p.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()
    # 设置PyTorch随机种子
    torch.manual_seed(args.seed)
    # 设置NumPy随机种子
    np.random.seed(args.seed)

    # 检测可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # dataset
    # 数据集来自于命令输入
    # 比如：
    """
        读取改位置中的所有npy文件用于训练
        ./unitree_rl_lab.sh -r \
          --data_dir /home/qihang/code_lab/unitree_rl_lab-main/logs/rsl_rl/collection_data/rec_cam_2026-01-10_19-07-45 \
          --epochs 50 \
          --batch_size 128 \
          --arch resnet18
    """
    ds = CamrecFolderDataset(args.data_dir, use_log2=args.use_log2, log_eps=args.log_eps)
    print(f"[INFO] loaded samples: {len(ds)} | label dim: {ds.out_dim}")

    # 计算验证集大小
    val_size = int(len(ds) * args.val_ratio)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # model
    # 模型初始化
    model = Depth2RayResNet(out_dim=ds.out_dim, arch=args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()# 创建均方误差损失函数

    # save/log dirs
    # os.makedirs(args.save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}_{args.arch}_N{len(ds)}"
    if args.exp_name:
        run_name += f"_{args.exp_name}"

    # 统一的实验目录
    run_dir = os.path.join(args.save_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = os.path.join(run_dir, "runs")
    writer = SummaryWriter(log_dir=tb_dir)

    print("[INFO] Ray prediction run dir:", os.path.abspath(run_dir))
    print("[INFO] TensorBoard log dir:", tb_dir)

    # training
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            global_step += 1

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
        val_loss /= max(1, len(val_loader))

        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        writer.add_scalar("loss/val_epoch", val_loss, epoch)

        print(f"[EPOCH {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(run_dir, "best.pt")
            torch.jit.script(model.to("cpu").eval()).save(best_path)
            model.to(device).train()
            print(f"[SAVE] best -> {best_path}")

        # periodic save
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(run_dir, f"epoch{epoch}.pt")
            torch.jit.script(model.to("cpu").eval()).save(ckpt_path)
            model.to(device).train()
            print(f"[SAVE] epoch -> {ckpt_path}")

    writer.close()
    print("[DONE]")

if __name__ == "__main__":
    main()
