import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from tqdm import tqdm


# --- 1. 数据集定义 (第一版原始逻辑: 无归一化, 纯物理数值) ---
class CamrecFolderDataset(Dataset):
    def __init__(self, rec_dir: str):
        self.rec_dir = rec_dir

        # 加载标签
        label_path = os.path.join(rec_dir, "label.pkl")
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"找不到标签文件: {label_path}")

        with open(label_path, "rb") as f:
            self.labels = pickle.load(f)

        # 获取目录下所有npy文件
        npy_files = [p for p in os.listdir(rec_dir) if p.endswith(".npy")]
        npy_files.sort()

        self.samples = []
        self.out_dim = None
        for fn in npy_files:
            key = os.path.splitext(fn)[0]
            if key in self.labels:
                y = np.asarray(self.labels[key], dtype=np.float32).reshape(-1)
                if self.out_dim is None: self.out_dim = y.shape[0]
                self.samples.append((os.path.join(rec_dir, fn), y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        depth = np.load(path).astype(np.float32)
        if depth.ndim == 3: depth = depth[..., 0]

        # 第一版核心：不做任何除法归一化，保持原始数值
        x = torch.from_numpy(depth).unsqueeze(0)
        y = torch.from_numpy(y)
        return x, y


# --- 2. 网络定义 (支持 ResNet18/34 切换) ---
class SimpleResNet(nn.Module):
    def __init__(self, out_dim, arch="resnet18"):
        super().__init__()
        # 根据参数选择架构
        if arch == "resnet34":
            self.net = models.resnet34(weights=None)
        else:
            self.net = models.resnet18(weights=None)

        # 替换全连接层
        self.net.fc = nn.Linear(self.net.fc.in_features, out_dim)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # 1通道转3通道
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    # --- 核心修复：把这些参数加回来，匹配你的命令 ---
    parser.add_argument("--data_dir", type=str, required=True, help="数据集路径")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34"], help="网络架构")

    args = parser.parse_args()

    # --- 3. 保存路径设置 (按你的要求) ---
    save_root = "logs/rsl_rl/ray_prediction"  # 注意：为了防止权限问题，建议加上 logs/ 前缀或绝对路径
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # 最终保存目录：logs/rsl_rl/ray_prediction/20260111-xxxxxx
    run_dir = os.path.join(save_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[INFO] 训练结果将保存至: {run_dir}")
    print(f"[INFO] 使用架构: {args.arch} | Batch Size: {args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    ds = CamrecFolderDataset(args.data_dir)
    print(f"[INFO] 加载样本数: {len(ds)} | 输出维度: {ds.out_dim}")

    train_size = int(len(ds) * 0.9)
    train_ds, val_ds = random_split(ds, [train_size, len(ds) - train_size])

    # 使用 args.batch_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)

    # 初始化模型
    model = SimpleResNet(ds.out_dim, arch=args.arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        # 训练循环
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- 4. 保存逻辑 ---
        # 保存 Best
        if val_loss < best_val:
            best_val = val_loss
            # TorchScript 保存
            torch.jit.script(model.to("cpu").eval()).save(os.path.join(run_dir, "best.pt"))
            model.to(device).train()
            print(f"  >>> [SAVE] Best model updated (Val: {val_loss:.4f})")

    # 保存 Final
    torch.jit.script(model.to("cpu").eval()).save(os.path.join(run_dir, "final.pt"))
    print(f"[DONE] 训练完成。模型保存在: {run_dir}")


if __name__ == "__main__":
    main()