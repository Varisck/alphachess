"""Neural network for AlphaZero-style chess

AlphaChessNet:
  - Starting Conv:   Conv2d + BN + ReLU
  - Tower:           Residual blocks x num_blocks
  - Policy head:     Conv: [B, 4672] logits
  - Value head:      Conv: scalar [-1, 1]
"""

from __future__ import annotations
from typing import Callable

import torch
import torch.nn as nn

from alphachess.storage import Storage

from collections import OrderedDict


class _ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(
            self, channels: int, 
            activation: Callable[[torch.Tensor], 
                                 torch.Tensor] = nn.functional.relu,
            kernel_size: int = 3, 
            padding: int = 1
            ):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias = False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float32 tensor of shape [B, channels, W, H]
        Returns:
            x: float32 tensor of shape [B, channels, W, H]
        """
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = res + x
        x = self.activation(x)
        return x

        


class AlphaChessNet(nn.Module):
    """AlphaChess Network"""

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        input_planes: int,
        padding: int = 1,
        kernel_size: int = 3,
        activation: Callable[[torch.Tensor], 
                             torch.Tensor] = nn.functional.relu,
        action_space: int = 4672
    ):
        super().__init__()
        self.action_space = action_space

        # First convs
        self.start_conv = nn.Conv2d(input_planes, channels, kernel_size,
                                    padding=padding, bias = False)
        self.start_bn = nn.BatchNorm2d(channels)
        self.activation = activation


        # tower: 
        self.tower = nn.Sequential(OrderedDict(
            {f"block_{i}" : _ResidualBlock(channels, activation, 
                                           kernel_size, padding) 
                                           for i in range(num_blocks)}
            ))

        # Policy head: output [8, 8, 73] (move encoding)
        self.policy_head = nn.Conv2d(channels, 73, 1)

        # Value head: output scalr [-1, 1]
        self.value_head = nn.Sequential(OrderedDict(
            {
                "value_head_conv" : nn.Conv2d(channels, 1, 1),
                "value_head_bn": nn.BatchNorm2d(1),
                "value_head_act": nn.ReLU(),
                "value_head_flatten": nn.Flatten(),
                "value_head_linear": nn.Linear(64, 256),
                "value_head_act2": nn.ReLU(),
                "value_head_linear2": nn.Linear(256, 1),
                "value_head_value_activation": nn.Tanh()
            }))


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: float32 tensor of shape [B, input_planes, 8, 8]
        Returns:
            policy_logits: [B, 4672]
            value:         [B, 1]  in [-1, 1]
        """
        x = self.start_conv(x)
        x = self.start_bn(x) 
        x = self.activation(x)

        x = self.tower(x)

        policy = self.policy_head(x).flatten(1)
        score = self.value_head(x)
        return (policy, score)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def save_to(self, storage: Storage, generation: int) -> None:
        """Atomically write checkpoint to models/{generation:006d}.pt."""
        import io
        buf = io.BytesIO()
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "num_blocks":   len(self.tower),
                "channels":     self.start_bn.num_features,
                "input_planes": self.start_conv.in_channels,
                "action_space": self.action_space,
            }
        }, buf)
        storage.atomic_put(f"models/{generation:06d}.pt", buf.getvalue())

    @classmethod
    def load_from(cls, storage: Storage, generation: int) -> "AlphaChessNet":
        """Load checkpoint written by save_to."""
        import io
        data = torch.load(
            io.BytesIO(storage.read_bytes(f"models/{generation:06d}.pt")),
            weights_only=False,
        )
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        return model