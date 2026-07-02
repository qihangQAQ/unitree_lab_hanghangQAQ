"""Height-scan FDM model for offline training on collected Unitree rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class OfflineFDMConfig:
    """Configuration for the offline FDM model."""

    state_dim: int
    proprio_dim: int
    height_scan_shape: tuple[int, int]
    prediction_horizon: int = 10
    history_length: int = 10
    command_timestep: float = 0.5
    action_dim: int = 3
    state_encoder_hidden: int = 64
    state_encoder_layers: int = 2
    action_encoder_dim: int = 16
    recurrence_hidden: int = 128
    recurrence_layers: int = 2
    dropout: float = 0.2
    collision_threshold: float = 0.5
    zero_collision_actions: bool = False
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "collision": 2.0,
            "position": 1.7,
            "heading": 1.7,
            "stop": 1.0,
            "energy": 0.0,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["height_scan_shape"] = list(self.height_scan_shape)
        return data


class MLP(nn.Module):
    """MLP matching the original FDM architecture exactly.

    Key behaviours inherited from FDM:
    - Single-layer (hidden=None): Linear -> LeakyReLU -> Dropout
    - Multi-layer: first Linear+Activation has NO dropout; subsequent hidden layers get Dropout.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden: list[int] | None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or []
        layers: list[nn.Module] = []
        if hidden:
            # first hidden layer: Linear + Activation, NO dropout (matching original FDM)
            layers.append(nn.Linear(input_dim, hidden[0]))
            layers.append(nn.LeakyReLU())
            # remaining hidden layers: Linear + Activation + optional Dropout
            for idx in range(len(hidden) - 1):
                layers.append(nn.Linear(hidden[idx], hidden[idx + 1]))
                layers.append(nn.LeakyReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
            # output layer (no activation)
            layers.append(nn.Linear(hidden[-1], output_dim))
        else:
            # single-layer: Linear + Activation + optional Dropout (original FDM treats
            # this as a hidden-like transformation, e.g. the action encoder)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LeakyReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.net.apply(self._init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _init(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=2**0.5)
            nn.init.zeros_(module.bias)


class HeightScanEncoder(nn.Module):
    """CNN encoder for the FDM 2-D height scan."""

    def __init__(self, height_scan_shape: tuple[int, int]):
        super().__init__()
        channels = [1, 32, 64, 128, 256]
        strides = [1, 2, 2, 2]
        kernels = [7, 3, 3, 3]
        layers: list[nn.Module] = []
        for idx in range(4):
            layers.append(
                nn.Conv2d(channels[idx], channels[idx + 1], kernel_size=kernels[idx], stride=strides[idx])
            )
            layers.append(nn.LeakyReLU())
            if idx == 0:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)
        self.net.apply(self._init)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *height_scan_shape)
            self.output_dim = int(self.net(dummy).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.net(x.float())

    @staticmethod
    def _init(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            # bias left at PyTorch default (matching original FDM)


class EmpiricalNormalization(nn.Module):
    """Online empirical normalizer for proprioceptive observations.

    Matching original FDM: supports an optional ``until`` parameter that freezes
    statistics after a given number of samples have been observed.
    """

    def __init__(self, dim: int, eps: float = 1e-2, until: int | None = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.until is not None and self.count >= self.until:
            return
        x = x.reshape(-1, x.shape[-1])
        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var.clamp_min(1e-8))


class OfflineFDMModel(nn.Module):
    """FDM height-scan multi-step velocity residual model."""

    def __init__(self, cfg: OfflineFDMConfig, device: str | torch.device = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device_name = str(device)
        self.proprioceptive_normalizer = EmpiricalNormalization(cfg.proprio_dim)
        self.state_obs_proprioceptive_encoder = nn.GRU(
            input_size=cfg.state_dim + cfg.proprio_dim,
            hidden_size=cfg.state_encoder_hidden,
            num_layers=cfg.state_encoder_layers,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.obs_exteroceptive_encoder = HeightScanEncoder(cfg.height_scan_shape)
        self.action_encoder = MLP(cfg.action_dim, cfg.action_encoder_dim, hidden=None, dropout=cfg.dropout)
        self.friction_predictor = MLP(cfg.state_encoder_hidden, 4, hidden=[32], dropout=0.0)

        recurrence_input = (
            cfg.action_encoder_dim + cfg.state_encoder_hidden + self.obs_exteroceptive_encoder.output_dim + 4
        )
        self.recurrence = nn.GRU(
            input_size=recurrence_input,
            hidden_size=cfg.recurrence_hidden,
            num_layers=cfg.recurrence_layers,
            dropout=cfg.dropout,
            batch_first=True,
        )
        head_input = cfg.recurrence_hidden * cfg.prediction_horizon
        self.state_predictor = MLP(head_input, cfg.prediction_horizon * 3, hidden=[128, 64], dropout=cfg.dropout)
        self.collision_predictor = MLP(head_input, cfg.prediction_horizon, hidden=[64], dropout=cfg.dropout)
        self.energy_predictor = MLP(head_input, cfg.prediction_horizon, hidden=[64], dropout=cfg.dropout)
        self.sigmoid = nn.Sigmoid()

        self.probability_loss = nn.BCELoss()
        self.position_loss = nn.MSELoss()
        self.heading_loss = nn.MSELoss()
        self.stop_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()

        # Match original FDM GRU initialisation: set update-gate bias to 1 so the
        # recurrent units start with a bias towards remembering past information.
        for gru in [self.state_obs_proprioceptive_encoder, self.recurrence]:
            for names in gru._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(gru, name)
                    n = bias.size(0)
                    start, end = n // 3, n // 3 * 2
                    bias.data[start:end].fill_(1.0)

        self.to(device)

    def forward(
        self, model_in: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, obs_proprioceptive, obs_exteroceptive, actions, _ = [x.to(self.device) for x in model_in]
        batch_size, traj_len, action_dim = actions.shape

        obs_proprioceptive = self.proprioceptive_normalizer(obs_proprioceptive)
        state_prop = torch.cat([state, obs_proprioceptive], dim=-1)
        encoded_state_prop = self.state_obs_proprioceptive_encoder(state_prop)[0][:, -1, :]
        encoded_height = self.obs_exteroceptive_encoder(obs_exteroceptive)
        encoded_actions = self.action_encoder(actions.reshape(-1, action_dim)).reshape(batch_size, traj_len, -1)
        friction = self.friction_predictor(encoded_state_prop)

        encoded_context = torch.cat([encoded_state_prop, encoded_height, friction], dim=-1)
        encoded_context = encoded_context.unsqueeze(1).repeat(1, traj_len, 1)
        recurrent_in = torch.cat([encoded_actions, encoded_context], dim=-1)
        recurrent_out = self.recurrence(recurrent_in)[0].reshape(batch_size, -1)

        collision_prob = self.sigmoid(self.collision_predictor(recurrent_out)).reshape(batch_size, traj_len)
        corr_vel = self.state_predictor(recurrent_out).reshape(batch_size, traj_len, 3)
        if self.cfg.zero_collision_actions:
            corr_vel = torch.where(collision_prob.unsqueeze(-1) > self.cfg.collision_threshold, -actions, corr_vel)
        corr_vel = corr_vel + actions

        corr_distance = corr_vel * self.cfg.command_timestep
        cumulative_yaw = corr_distance[..., 2].cumsum(dim=1)
        rot = torch.stack(
            [
                torch.stack([torch.cos(cumulative_yaw), -torch.sin(cumulative_yaw)], dim=-1),
                torch.stack([torch.sin(cumulative_yaw), torch.cos(cumulative_yaw)], dim=-1),
            ],
            dim=-2,
        )
        rot = torch.roll(rot, shifts=1, dims=1)
        rot[:, 0] = torch.eye(2, device=rot.device, dtype=rot.dtype)
        local_steps = torch.matmul(rot, corr_distance[..., :2].unsqueeze(-1)).squeeze(-1)
        cumulative_pos = local_steps.cumsum(dim=1)
        state_traj = torch.cat(
            [cumulative_pos, torch.sin(cumulative_yaw).unsqueeze(-1), torch.cos(cumulative_yaw).unsqueeze(-1)],
            dim=-1,
        )

        energy = self.energy_predictor(recurrent_out).reshape(batch_size, traj_len, 1)
        return state_traj, collision_prob, energy

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def loss(
        self, model_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor, mode: str = "train"
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_state, pred_collision, pred_energy = model_out
        target = target.to(self.device)
        target_state = target[..., :4]
        target_collision = target[..., 4]
        target_energy = target[..., 5] if target.shape[-1] > 5 else torch.zeros_like(target_collision)

        collision_loss = self.probability_loss(pred_collision, target_collision)
        position_loss = sum(
            self.position_loss(pred_state[:, idx, :2], target_state[:, idx, :2])
            for idx in range(pred_state.shape[1])
        )
        heading_loss = sum(
            self.heading_loss(pred_state[:, idx, 2:], target_state[:, idx, 2:])
            for idx in range(pred_state.shape[1])
        )
        stop_loss = self._stop_loss(pred_state, target_collision, target_state)
        energy_loss = self.energy_loss(pred_energy.squeeze(-1), target_energy)

        weights = self.cfg.loss_weights
        loss = (
            weights["collision"] * collision_loss
            + weights["position"] * position_loss
            + weights["heading"] * heading_loss
            + weights["stop"] * stop_loss
            + weights["energy"] * energy_loss
        )
        meta = {
            f"{mode}_loss": float(loss.detach().cpu()),
            f"{mode}_collision_loss": float(collision_loss.detach().cpu()),
            f"{mode}_position_loss": float(position_loss.detach().cpu()),
            f"{mode}_heading_loss": float(heading_loss.detach().cpu()),
            f"{mode}_stop_loss": float(stop_loss.detach().cpu()),
            f"{mode}_energy_loss": float(energy_loss.detach().cpu()),
        }
        return loss, meta

    def update(
        self,
        model_in: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, dict[str, float]]:
        self.train()
        optimizer.zero_grad()
        loss, meta = self.loss(self.forward(model_in), target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 2.0)
        optimizer.step()
        return float(loss.detach().cpu()), meta

    @torch.no_grad()
    def evaluate_batch(
        self,
        model_in: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
        self.eval()
        model_out = self.forward(model_in)
        loss, meta = self.loss(model_out, target, mode="val")
        pred_collision = model_out[1]
        target_collision = target.to(self.device)[..., 4]
        pred_binary = pred_collision > self.cfg.collision_threshold
        target_binary = target_collision > 0.5
        meta["val_collision_accuracy"] = float((pred_binary == target_binary).float().mean().detach().cpu())
        return float(loss.detach().cpu()), meta

    def save(self, path: str | Path, meta: dict[str, Any] | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "cfg": self.cfg.to_dict(),
                "meta": meta or {},
            },
            path,
        )

    @staticmethod
    def _stop_loss(
        pred_state: torch.Tensor, target_collision: torch.Tensor, target_state: torch.Tensor
    ) -> torch.Tensor:
        mask = target_collision == 1
        if not torch.any(mask):
            return pred_state.new_tensor(0.0)
        return nn.functional.mse_loss(pred_state[mask], target_state[mask])
