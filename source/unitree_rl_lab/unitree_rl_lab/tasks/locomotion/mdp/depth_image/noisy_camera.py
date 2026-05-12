from __future__ import annotations

import inspect
import torch
from prettytable import PrettyTable
from typing import Sequence

from isaaclab.utils import string_to_callable

from .utils.async_circular_buffer import AsyncCircularBuffer
from .utils.noise_cfg import ImageNoiseCfg


class NoisyCameraMixin:
    def __str__(self) -> str:
        return_ = super().__str__()
        noise_info_table = PrettyTable()
        noise_info_table.field_names = ["Noise Name", "Noise Cfg Name"]
        for noise_name, noise_cfg in self.cfg.noise_pipeline.items():
            noise_info_table.add_row([noise_name, noise_cfg.__name__])
        return_ += "\n" + str(noise_info_table)
        history_info_table = PrettyTable()
        history_info_table.field_names = ["History Name", "History Length"]
        for history_name, history_length in self.cfg.data_histories.items():
            history_info_table.add_row([history_name, history_length])
        return_ += "\n" + str(history_info_table)
        return return_

    def build_noise_pipeline(self):
        self.noise_pipeline: Sequence[ImageNoiseCfg] | list[ImageNoiseCfg] = []

        for noise_name, noise_cfg in self.cfg.noise_pipeline.items():
            if not isinstance(noise_cfg, ImageNoiseCfg):
                raise ValueError(f"Invalid noise configuration for {noise_name}: {noise_cfg}")

            noise_cfg.device = self.device

            if isinstance(noise_cfg.func, str):
                noise_cfg.func = string_to_callable(noise_cfg.func)

            if inspect.isclass(noise_cfg.func):
                noise_cfg.func = noise_cfg.func(noise_cfg, num_envs=self.num_instances, device=self.device)

            self.noise_pipeline.append(noise_cfg)

        for data_type in self.cfg.data_types:
            self._data.output[f"{data_type}_noised"] = self.apply_noise_pipeline(
                self._data.output[data_type], env_ids=self._ALL_INDICES
            )

    def apply_noise_pipeline(self, data: torch.Tensor, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        data = data.clone()
        for noise_cfg in self.noise_pipeline:
            data = noise_cfg.func(data, noise_cfg, env_ids)

        return data

    def apply_noise_pipeline_to_all_data_types(self, env_ids: torch.Tensor | Sequence[int]):
        for data_type in self.cfg.data_types:
            self._data.output[f"{data_type}_noised"][env_ids] = self.apply_noise_pipeline(
                self._data.output[data_type][env_ids], env_ids=env_ids
            )

    def reset_noise_pipeline(self, env_ids: Sequence[int] | None = None):
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        for noise_cfg in self.noise_pipeline:
            if hasattr(noise_cfg.func, "reset"):
                noise_cfg.func.reset(env_ids)

    def build_history_buffers(self):
        self.output_history_buffers: dict[str, AsyncCircularBuffer] = dict()

        for data_type, history_length in self.cfg.data_histories.items():
            self.output_history_buffers[data_type] = AsyncCircularBuffer(
                history_length, self.num_instances, self.device
            )
            data_shape = self._data.output[data_type].shape
            self._data.output[f"{data_type}_history"] = torch.zeros(
                (data_shape[0], history_length, *data_shape[1:]), device=self.device
            )

    def update_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].append(self._data.output[data_type][env_ids], env_ids)
            self._data.output[f"{data_type}_history"][env_ids] = self.output_history_buffers[data_type].__getitem__(
                batch_ids=env_ids
            )

    def reset_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].reset(env_ids)
