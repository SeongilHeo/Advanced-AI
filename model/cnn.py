from typing import Callable, Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

ActivationFactory = Callable[[], nn.Module]
ParamLike = Union[int, Tuple[int, int], Sequence[Union[int, Tuple[int, int]]]]

activation_table: Dict[str, ActivationFactory] = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
}


def _expand_param(param: ParamLike, n_layers: int, name: str):
    if isinstance(param, int):
        return [param for _ in range(n_layers)]

    # 2D tuple (e.g., (k_h, k_w)) means "same for all layers".
    if isinstance(param, tuple) and len(param) == 2 and all(
        isinstance(v, int) for v in param
    ):
        return [param for _ in range(n_layers)]

    values = list(param)
    if len(values) != n_layers:
        raise ValueError(f"{name} length must match number of conv layers ({n_layers}).")
    return values


class CNN(nn.Module):
    """
    Configurable CNN for image observations.
    Default values match the classic Atari DQN-style backbone.
    """

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        input_hw: Tuple[int, int] = (84, 84),
        conv_channels: Sequence[int] = (32, 64, 64),
        conv_kernel_sizes: ParamLike = (8, 4, 3),
        conv_strides: ParamLike = (4, 2, 1),
        conv_paddings: ParamLike = 0,
        conv_activation: str = "relu",
        head_hidden_dim: int = 512,
        head_num_layers: int = 0,
        head_activation: str = "relu",
    ) -> None:
        super().__init__()

        if conv_activation not in activation_table:
            raise ValueError(f"Unknown conv_activation: {conv_activation}")
        if head_activation not in activation_table:
            raise ValueError(f"Unknown head_activation: {head_activation}")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one layer size.")

        n_conv_layers = len(conv_channels)
        kernel_sizes = _expand_param(conv_kernel_sizes, n_conv_layers, "conv_kernel_sizes")
        strides = _expand_param(conv_strides, n_conv_layers, "conv_strides")
        paddings = _expand_param(conv_paddings, n_conv_layers, "conv_paddings")

        conv_blocks = []
        prev_channels = int(in_channels)
        conv_act_factory = activation_table[conv_activation]
        for out_channels, kernel_size, stride, padding in zip(
            conv_channels,
            kernel_sizes,
            strides,
            paddings,
        ):
            conv_blocks.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=int(out_channels),
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            conv_blocks.append(conv_act_factory())
            prev_channels = int(out_channels)

        self.conv = nn.Sequential(*conv_blocks)

        # Infer conv output size with a dummy forward.
        with torch.no_grad():
            h, w = input_hw
            dummy = torch.zeros(1, in_channels, h, w)
            out = self.conv(dummy)
            conv_out_dim = int(np.prod(out.shape[1:]))

        head_layers = [nn.Flatten(), nn.Linear(conv_out_dim, head_hidden_dim)]
        head_act_factory = activation_table[head_activation]
        head_layers.append(head_act_factory())
        for _ in range(int(head_num_layers)):
            head_layers.append(nn.Linear(head_hidden_dim, head_hidden_dim))
            head_layers.append(head_act_factory())
        head_layers.append(nn.Linear(head_hidden_dim, n_actions))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x)
