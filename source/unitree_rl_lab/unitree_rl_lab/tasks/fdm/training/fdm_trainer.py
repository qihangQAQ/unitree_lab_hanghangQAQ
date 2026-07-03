"""Trainer for the Unitree FDM model."""

from __future__ import annotations

import datetime as _datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from unitree_rl_lab.tasks.fdm.model import FDMConfig, FDMModel
from unitree_rl_lab.tasks.fdm.training.fdm_dataset import FDMTrajectoryDataset, load_fdm_payload


@dataclass
class FDMTrainingConfig:
    dataset: str | None = None
    output_root: str = "logs/fdm/unitree_g1_fdm"
    run_name: str | None = None
    device: str = "cuda:0"
    prediction_horizon: int = 10
    num_samples: int = 80000
    batch_size: int = 512
    epochs: int = 8
    learning_rate: float = 3e-3
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 4
    seed: int = 0
    no_hard_contact_normalization: bool = False
    inspect_only: bool = False
    collision_rate: float | None = None
    no_collision_oversampling: bool = False
    sample_filter_first_steps_coll: int = 0
    small_motion_ratio: float | None = 0.1
    small_motion_threshold: float = 1.0
    height_threshold: float | None = None
    outlier_threshold: float = 10.0
    apply_noise: bool = False
    """Apply additive uniform noise to observations during training, matching reference FDM."""


class FDMTrainer:
    """Train an FDM model from collected rollout data."""

    def __init__(
        self,
        cfg: FDMTrainingConfig,
        payload: dict[str, Any] | None = None,
        val_payload: dict[str, Any] | None = None,
        model: FDMModel | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        log_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
        if payload is None:
            if cfg.dataset is None:
                raise ValueError("FDMTrainer needs either cfg.dataset or an in-memory payload.")
            payload = load_fdm_payload(cfg.dataset)
        val_payload = payload if val_payload is None else val_payload
        self.train_dataset = FDMTrajectoryDataset(
            payload,
            prediction_horizon=cfg.prediction_horizon,
            num_samples=cfg.num_samples,
            split="train",
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            device="cpu",
            normalize_hard_contact=not cfg.no_hard_contact_normalization,
            collision_rate=cfg.collision_rate,
            include_collision_samples=not cfg.no_collision_oversampling,
            sample_filter_first_steps_coll=cfg.sample_filter_first_steps_coll,
            small_motion_ratio=cfg.small_motion_ratio,
            small_motion_threshold=cfg.small_motion_threshold,
            height_threshold=cfg.height_threshold,
            outlier_threshold=cfg.outlier_threshold,
        )
        val_samples = max(1, int(cfg.num_samples * cfg.val_ratio / max(1.0 - cfg.val_ratio, 1e-6)))
        self.val_dataset = FDMTrajectoryDataset(
            val_payload,
            prediction_horizon=cfg.prediction_horizon,
            num_samples=val_samples,
            split="val",
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            device="cpu",
            normalize_hard_contact=not cfg.no_hard_contact_normalization,
            collision_rate=cfg.collision_rate,
            include_collision_samples=not cfg.no_collision_oversampling,
            sample_filter_first_steps_coll=cfg.sample_filter_first_steps_coll,
            small_motion_ratio=cfg.small_motion_ratio,
            small_motion_threshold=cfg.small_motion_threshold,
            height_threshold=cfg.height_threshold,
            outlier_threshold=cfg.outlier_threshold,
        )
        model_cfg = FDMConfig(
            state_dim=self.train_dataset.state_dim,
            proprio_dim=self.train_dataset.proprio_dim,
            height_scan_shape=self.train_dataset.height_scan_shape,
            prediction_horizon=cfg.prediction_horizon,
            history_length=self.train_dataset.history_length,
            command_timestep=self.train_dataset.command_timestep,
        )
        if model is None:
            self.model = FDMModel(model_cfg, device=self.device)
        else:
            self.model = model.to(self.device)
            if self.model.cfg.to_dict() != model_cfg.to_dict():
                raise ValueError("Existing FDM model config does not match the newly collected dataset dimensions.")
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=cfg.min_learning_rate
        )
        self.log_dir = Path(log_dir).expanduser().resolve() if log_dir is not None else self._make_log_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # ---- per-dim proprioceptive noise bounds (matching reference FDM) ----
        self._proprio_noise_min: torch.Tensor | None = None
        self._proprio_noise_max: torch.Tensor | None = None
        if cfg.apply_noise:
            self._proprio_noise_min, self._proprio_noise_max = self._build_proprio_noise_bounds()

    def train(self) -> dict[str, Any]:
        if self.cfg.inspect_only:
            self._save_config()
            sample = self.train_dataset[0]
            names = [
                "state_history",
                "proprioceptive",
                "exteroceptive",
                "actions",
                "add_exteroceptive",
                "target",
                "perfect_velocity",
            ]
            print("[INFO] Inspect-only dataset sample:")
            for name, value in zip(names, sample):
                print(f"  {name}: {tuple(value.shape)} {value.dtype}")
            print(f"[INFO] Train collision rate: {self.train_dataset.collision_rate:.4f}")
            print(f"[INFO] Train filter stats: {self.train_dataset.filter_stats}")
            print(f"[INFO] Val filter stats: {self.val_dataset.filter_stats}")
            return {"log_dir": str(self.log_dir), "best_val_loss": float("nan"), "history": []}

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self._save_config()
        best_val = float("inf")
        history: list[dict[str, float]] = []
        for epoch in range(self.cfg.epochs):
            train_loss = self._run_train_epoch(train_loader)
            val_loss, val_meta = self._run_val_epoch(val_loader)
            self.scheduler.step(val_loss)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_collision_accuracy": val_meta.get("val_collision_accuracy", 0.0),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"[INFO] Epoch {epoch:03d}: train={train_loss:.5f} val={val_loss:.5f} "
                f"acc={row['val_collision_accuracy']:.4f} lr={row['learning_rate']:.3e}"
            )
            if val_loss < best_val:
                best_val = val_loss
                self.model.save(self.log_dir / "model_best.pth", meta={"epoch": epoch, "val_loss": val_loss})
        self.model.save(self.log_dir / "model_last.pth", meta={"val_loss": val_loss})
        with open(self.log_dir / "losses.yaml", "w") as f:
            yaml.safe_dump({"history": history, "best_val_loss": best_val}, f)
        return {"log_dir": str(self.log_dir), "best_val_loss": best_val, "history": history}

    def _run_train_epoch(self, loader: DataLoader) -> float:
        losses = []
        for batch in loader:
            model_in, target, _ = self._prepare_batch(batch)
            loss, _ = self.model.update(model_in, target, self.optimizer)
            losses.append(loss)
        return float(sum(losses) / max(len(losses), 1))

    @torch.no_grad()
    def _run_val_epoch(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        losses = []
        meta_accum: dict[str, float] = {}
        for batch in loader:
            model_in, target, _ = self._prepare_batch(batch)
            loss, meta = self.model.evaluate_batch(model_in, target)
            losses.append(loss)
            for key, value in meta.items():
                meta_accum[key] = meta_accum.get(key, 0.0) + value
        for key in meta_accum:
            meta_accum[key] /= max(len(loader), 1)
        return float(sum(losses) / max(len(losses), 1)), meta_accum

    def _prepare_batch(self, batch) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        state_history, proprio, extero, actions, add_extero, target, perfect_velocity = batch

        # ---- observation noise (matching reference FDM, training only) ----
        if self.model.training and self._proprio_noise_min is not None:
            noise_min = self._proprio_noise_min.to(proprio.device)
            noise_max = self._proprio_noise_max.to(proprio.device)
            proprio = proprio + torch.empty_like(proprio).uniform_() * (noise_max - noise_min) + noise_min
            extero = extero + torch.empty_like(extero).uniform_(-0.01, 0.01)

        model_in = (
            state_history.to(self.device, non_blocking=True),
            proprio.to(self.device, non_blocking=True),
            extero.to(self.device, non_blocking=True),
            actions.to(self.device, non_blocking=True),
            add_extero.to(self.device, non_blocking=True),
        )
        return model_in, target.to(self.device, non_blocking=True), perfect_velocity.to(self.device, non_blocking=True)

    def _make_log_dir(self) -> Path:
        stamp = _datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{stamp}_{self.cfg.run_name}" if self.cfg.run_name else stamp
        log_dir = Path(self.cfg.output_root).expanduser().resolve() / name
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _build_proprio_noise_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-dim uniform noise bounds matching reference FDM noise levels.

        Proprioceptive layout (same for ANYmal-D and G1, differing only in joint count):
            [vel_cmd(3), gravity(3), base_lin_vel(3), base_ang_vel(3),
             joint_block_0, ..., joint_block_9]
        where each joint_block has n_joints entries and
        n_joints = (proprio_dim - 12) // 10.
        """
        pdim = self.train_dataset.proprio_dim
        n_joints = (pdim - 12) // 10
        noise_min = torch.zeros(pdim)
        noise_max = torch.zeros(pdim)

        # non-joint terms (first 12 dims)
        noise_min[0:3] = 0.0;   noise_max[0:3] = 0.0     # vel_cmd: no noise
        noise_min[3:6] = -0.05; noise_max[3:6] = 0.05     # projected_gravity
        noise_min[6:9] = -0.1;  noise_max[6:9] = 0.1      # base_lin_vel
        noise_min[9:12] = -0.2; noise_max[9:12] = 0.2     # base_ang_vel

        def _block(start: int) -> slice:
            return slice(start, start + n_joints)

        # block 0: joint_torque — no noise
        # block 1: joint_pos — ±0.01
        noise_min[_block(12 + 1 * n_joints)] = -0.01
        noise_max[_block(12 + 1 * n_joints)] = 0.01
        # block 2: joint_vel — ±1.5
        noise_min[_block(12 + 2 * n_joints)] = -1.5
        noise_max[_block(12 + 2 * n_joints)] = 1.5
        # blocks 3-5: joint_pos_error_idx{0,2,4} — ±0.01
        for b in (3, 4, 5):
            noise_min[_block(12 + b * n_joints)] = -0.01
            noise_max[_block(12 + b * n_joints)] = 0.01
        # blocks 6-7: joint_vel_idx{2,4} — ±1.5
        for b in (6, 7):
            noise_min[_block(12 + b * n_joints)] = -1.5
            noise_max[_block(12 + b * n_joints)] = 1.5
        # blocks 8-9: last_action, second_last_action — no noise

        return noise_min, noise_max

    def _save_config(self):
        params_dir = self.log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        with open(params_dir / "trainer_cfg.yaml", "w") as f:
            yaml.safe_dump(asdict(self.cfg), f)
        with open(params_dir / "model_cfg.yaml", "w") as f:
            yaml.safe_dump(self.model.cfg.to_dict(), f)
        dataset_meta = {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "state_dim": self.train_dataset.state_dim,
            "raw_state_dim": self.train_dataset.state_dim_raw,
            "proprio_dim": self.train_dataset.proprio_dim,
            "height_scan_shape": list(self.train_dataset.height_scan_shape),
            "collision_rate": self.train_dataset.collision_rate,
            "train_filter_stats": self.train_dataset.filter_stats,
            "val_filter_stats": self.val_dataset.filter_stats,
            "source_meta": self.train_dataset.meta,
        }
        with open(params_dir / "dataset_meta.yaml", "w") as f:
            yaml.safe_dump(dataset_meta, f)
