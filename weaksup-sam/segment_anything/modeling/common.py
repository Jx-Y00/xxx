# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math

from typing import Type


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class vpAdapter(nn.Module):
    def __init__(self,
                 embedding_dim,
                 adapter_bn: int = 64,
                 adapter_act: Type[nn.Module] = nn.GELU,
                 adapter_dropout: float = 0.1,
                 adapter_scalar: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter_down = nn.Linear(embedding_dim, adapter_bn)
        self.adapter_up = nn.Linear(adapter_bn, embedding_dim)
        self.adapter_act = adapter_act()
        self.adapter_dropout = adapter_dropout
        if adapter_scalar is not None:
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = adapter_scalar

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, vp: torch.Tensor) -> torch.Tensor:
        adapt = self.adapter_up(nn.functional.dropout(
            self.adapter_act(self.adapter_down(vp)),
            p=self.adapter_dropout, training=self.training))
        return adapt * self.adapter_scale


if __name__ == "__main__":
    input_tensor = torch.randn((2, 3, 3, 768))
    mlp_block = MLPBlock(embedding_dim=768, mlp_dim=64)
    adapter_block = vpAdapter(mlp=mlp_block)
    output_tensor = mlp_block(input_tensor)
    out = adapter_block(input_tensor)
    print(output_tensor.shape)
    print(out.shape)
