# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import annotations

"""Construction hooks and existing TP and fused-shard checkpoint loading."""

import pytest
import torch

from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config


class _CustomFp8Method(Fp8LinearMethod):
    def create_weights(self, layer, *args, **kwargs):
        super().create_weights(layer, *args, **kwargs)
        # Observable construction behavior; later loading must overwrite it.
        layer.weight.data.fill_(7)


class _CustomMxfp8Config(Mxfp8Config):
    def get_quant_method(self, layer, prefix):
        assert prefix == "model.proj"
        assert layer.input_size == 128
        assert not list(layer.parameters())
        return _CustomFp8Method(self)


def _make_layer(kind, rank, config):
    common = dict(
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=config,
        prefix="model.proj",
    )
    parallel = dict(tp_rank=rank, tp_size=2, tp_group=(0, 1))
    if kind == "replicated":
        return ReplicatedLinear(128, 128, **common)
    if kind == "column":
        return ColumnParallelLinear(128, 128, **common, **parallel)
    if kind == "merged":
        return MergedColumnParallelLinear(128, [64, 128], **common, **parallel)
    if kind == "qkv":
        # Q is sharded; the single KV head is replicated across both ranks.
        return QKVParallelLinear(128, 32, 4, 1, **common, **parallel)
    if kind == "row":
        return RowParallelLinear(128, 128, **common, **parallel)
    raise AssertionError(f"Unknown linear kind: {kind}")


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind", ["replicated", "column", "merged", "qkv", "row"])
@pytest.mark.parametrize("custom", [False, True])
def test_fp8_construction_and_checkpoint_sharding(kind, rank, custom):
    config_cls = _CustomMxfp8Config if custom else Mxfp8Config
    config = config_cls(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
        scale_fmt="ue8m0",
    )
    layer = _make_layer(kind, rank, config)
    if custom:
        assert torch.all(layer.weight.float() == 7)

    if kind == "merged":
        pieces = [(0, 64), (1, 128)]
    elif kind == "qkv":
        pieces = [("q", 128), ("k", 32), ("v", 32)]
    else:
        pieces = [(None, 128)]
    expected_weights = []
    expected_scales = []
    for index, (shard, rows) in enumerate(pieces):
        weight = (torch.arange(rows * 128).reshape(rows, 128) % 31) - 15
        weight = (weight + index).to(torch.float8_e4m3fn)
        scale = (torch.arange(rows * 4).reshape(rows, 4) % 13 + 120 + index).to(
            torch.uint8
        )
        args = () if shard is None else (shard,)
        layer.weight.weight_loader(layer.weight, weight, *args)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale, *args)
        if kind == "row":
            weight = weight[:, rank * 64 : (rank + 1) * 64]
            scale = scale[:, rank * 2 : (rank + 1) * 2]
        elif kind in ("column", "merged") or (kind == "qkv" and shard == "q"):
            weight = weight[rank * (rows // 2) : (rank + 1) * (rows // 2)]
            scale = scale[rank * (rows // 2) : (rank + 1) * (rows // 2)]
        expected_weights.append(weight.float())
        expected_scales.append(scale)
    torch.testing.assert_close(layer.weight.float(), torch.cat(expected_weights))
    torch.testing.assert_close(layer.weight_scale_inv, torch.cat(expected_scales))


@pytest.mark.parametrize("config_cls", [Fp8Config, Mxfp8Config])
@pytest.mark.parametrize("block", [None, [32, 32]])
def test_existing_fp8_parameter_formats(config_cls, block):
    config = config_cls(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=block,
    )
    layer = ReplicatedLinear(128, 64, bias=False, quant_config=config)
    assert type(layer.quant_method) is Fp8LinearMethod
    assert layer.weight.dtype == torch.float8_e4m3fn
    if block is None:
        assert layer.weight_scale.shape == (1,)
        assert layer.weight_scale.dtype == torch.float32
    else:
        assert layer.weight_scale_inv.shape == (2, 4)
        assert layer.weight_scale_inv.dtype == config.weight_scale_dtype


@pytest.mark.parametrize("ignored", [["model.proj"], ["re:.*proj"]])
def test_ignored_fp8_layer_does_not_call_custom_factory(ignored):
    config = _CustomMxfp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=ignored,
        weight_block_size=[1, 32],
    )
    layer = _make_layer("replicated", 0, config)
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert layer.weight.dtype == torch.bfloat16


def test_fused_member_exclusion_precedes_custom_factory():
    config = _CustomMxfp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=["model.key_proj", "model.value_proj"],
        weight_block_size=[1, 32],
    )
    layer = ReplicatedLinear(128, 128, quant_config=config, prefix="model.kv_proj")
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
