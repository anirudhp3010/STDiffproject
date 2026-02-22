"""
Euler ODE sampler for flow-matching / v-prediction.

Integrates dx/dt = v(x, t) from t=1 (noise) to t=0 (data) using Euler method.
Compatible with diffusers pipeline interface: set_timesteps, timesteps, step().
Inherits from SchedulerMixin so pipeline.save_pretrained() works without transformers.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class EulerFlowSchedulerOutput:
    """Output of EulerFlowScheduler.step(). Compatible with scheduler output."""

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class EulerFlowScheduler(SchedulerMixin):
    """
    Euler integrator for flow-matching ODE: dx/dt = v(x, t).

    Integrates backward from t=1 (noise) to t=0 (data).
    Model predicts velocity v directly; no epsilon/sample conversion.
    """

    def __init__(
        self,
        num_inference_steps: int = 50,
        device: Optional[torch.device] = None,
    ):
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.timesteps = None
        self._all_timesteps = None  # includes final t=0 for dt computation

    def set_timesteps(self, num_inference_steps: Optional[int] = None, device: Optional[torch.device] = None):
        """Set timesteps for integration: t from 1 (noise) down to 0 (data)."""
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
        if device is not None:
            self.device = device

        # t_steps: [1.0, ..., 0.0] (num_inference_steps + 1 values)
        self._all_timesteps = torch.linspace(
            1.0, 0.0, self.num_inference_steps + 1, dtype=torch.float32
        )
        if self.device is not None:
            self._all_timesteps = self._all_timesteps.to(self.device)

        # Timesteps to iterate: all except the last (we step from each to the next)
        self.timesteps = self._all_timesteps[:-1]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[EulerFlowSchedulerOutput, Tuple[torch.Tensor]]:
        """
        Euler step: x_next = x_cur + (t_next - t_cur) * v_pred.

        We integrate backward (t decreases), so t_next < t_cur and dt < 0.

        Args:
            model_output: Predicted velocity v(x_t, t). Same shape as sample.
            timestep: Current time t (float in [0, 1] or tensor).
            sample: Current state x_t.

        Returns:
            prev_sample: x_{t_next} = x_t + (t_next - t) * v
        """
        if self._all_timesteps is None:
            self.set_timesteps(device=sample.device)

        # Ensure timestep is on same device as sample
        if torch.is_tensor(timestep):
            t_cur = timestep.float().to(sample.device)
            if t_cur.dim() == 0:
                t_cur = t_cur.unsqueeze(0)
        else:
            t_cur = torch.tensor(float(timestep), device=sample.device, dtype=sample.dtype)

        # Find index of current timestep: t goes from 1 to 0, so idx 0 corresponds to t=1
        t_cur_val = t_cur.item() if t_cur.numel() == 1 else t_cur[0].item()
        # idx such that _all_timesteps[idx] ~= t_cur
        idx = int(round((1.0 - t_cur_val) * self.num_inference_steps))
        idx = max(0, min(idx, self.num_inference_steps - 1))

        if idx >= len(self._all_timesteps) - 1:
            t_next = torch.tensor(0.0, device=sample.device, dtype=sample.dtype)
        else:
            t_next = self._all_timesteps[idx + 1].to(sample.device)

        # dt = t_next - t_cur (negative when going from 1 to 0)
        dt = t_next - t_cur
        if dt.dim() < sample.dim():
            dt = dt.reshape((-1,) + (1,) * (sample.dim() - 1))

        prev_sample = sample + dt * model_output

        if not return_dict:
            return (prev_sample,)
        return EulerFlowSchedulerOutput(prev_sample=prev_sample)

    @property
    def config(self) -> dict:
        """Config-like dict for compatibility (e.g. prediction_type check)."""
        return {"prediction_type": "v"}

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Save scheduler config for loading at inference."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        config = {
            "scheduler_type": "EulerFlow",
            "prediction_type": "v",
            "num_inference_steps": self.num_inference_steps,
        }
        with open(save_path / "scheduler_config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path], **kwargs) -> "EulerFlowScheduler":
        """Load scheduler config from directory."""
        save_path = Path(save_directory)
        config_path = save_path / "scheduler_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return cls(
                num_inference_steps=config.get("num_inference_steps", 50),
                **kwargs,
            )
        return cls(num_inference_steps=50, **kwargs)
