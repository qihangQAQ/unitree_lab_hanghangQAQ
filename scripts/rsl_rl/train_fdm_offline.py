#!/usr/bin/env python3

"""Train the Unitree FDM network from collected rollout data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "unitree_rl_lab"))

from unitree_rl_lab.tasks.fdm.training import OfflineFDMTrainer, OfflineTrainingConfig


def parse_args() -> OfflineTrainingConfig:
    parser = argparse.ArgumentParser(description="Offline FDM training from collected trajectories.")
    parser.add_argument("--dataset", required=True, help="Path to fdm_trajectories.pkl from collect_fdm_rollouts.py.")
    parser.add_argument("--output_root", default="logs/fdm/unitree_g1_fdm")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction_horizon", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=80000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--min_learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_hard_contact_normalization", action="store_true", default=False)
    parser.add_argument("--inspect_only", action="store_true", default=False)
    parser.add_argument("--collision_rate", type=float, default=None)
    parser.add_argument("--no_collision_oversampling", action="store_true", default=False)
    parser.add_argument("--sample_filter_first_steps_coll", type=int, default=0)
    parser.add_argument("--small_motion_ratio", type=float, default=0.1)
    parser.add_argument("--small_motion_threshold", type=float, default=1.0)
    parser.add_argument("--height_threshold", type=float, default=None)
    parser.add_argument("--outlier_threshold", type=float, default=10.0)
    args = parser.parse_args()
    return OfflineTrainingConfig(**vars(args))


def main():
    cfg = parse_args()
    trainer = OfflineFDMTrainer(cfg)
    print(f"[INFO] Training samples: {len(trainer.train_dataset)}")
    print(f"[INFO] Validation samples: {len(trainer.val_dataset)}")
    print(f"[INFO] State dim: {trainer.train_dataset.state_dim}, proprio dim: {trainer.train_dataset.proprio_dim}")
    print(f"[INFO] Height scan: {trainer.train_dataset.height_scan_shape}")
    result = trainer.train()
    print(f"[INFO] Done. Best val loss: {result['best_val_loss']:.6f}")
    print(f"[INFO] Log dir: {result['log_dir']}")


if __name__ == "__main__":
    main()
