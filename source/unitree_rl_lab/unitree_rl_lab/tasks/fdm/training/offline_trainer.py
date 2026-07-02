"""Offline trainer for the Unitree FDM model."""

from __future__ import annotations

import datetime as _datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from unitree_rl_lab.tasks.fdm.model import OfflineFDMConfig, OfflineFDMModel
from unitree_rl_lab.tasks.fdm.training.offline_dataset import OfflineFDMTrajectoryDataset, load_fdm_payload


@dataclass
class OfflineTrainingConfig:
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


class OfflineFDMTrainer:
    """Train an FDM model from one collected rollout pickle."""

    def __init__(
        self,
        cfg: OfflineTrainingConfig,
        payload: dict[str, Any] | None = None,
        val_payload: dict[str, Any] | None = None,
        model: OfflineFDMModel | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        log_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
        if payload is None:
            if cfg.dataset is None:
                raise ValueError("OfflineFDMTrainer needs either cfg.dataset or an in-memory payload.")
            payload = load_fdm_payload(cfg.dataset)
        val_payload = payload if val_payload is None else val_payload
        self.train_dataset = OfflineFDMTrajectoryDataset(
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
        self.val_dataset = OfflineFDMTrajectoryDataset(
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
        model_cfg = OfflineFDMConfig(
            state_dim=self.train_dataset.state_dim,
            proprio_dim=self.train_dataset.proprio_dim,
            height_scan_shape=self.train_dataset.height_scan_shape,
            prediction_horizon=cfg.prediction_horizon,
            history_length=self.train_dataset.history_length,
            command_timestep=self.train_dataset.command_timestep,
        )
        if model is None:
            self.model = OfflineFDMModel(model_cfg, device=self.device)
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
