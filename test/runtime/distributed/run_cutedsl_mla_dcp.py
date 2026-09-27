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

"""CuTe MLA DCP decode against unsharded attention, using real collectives.

Run with torchrun --standalone --nproc-per-node=2 (or 4) and this file.
Covers BF16/FP8, decode/verify/draft, empty shards, and eager/graph replay.
"""

import argparse
import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_backend.registry import initialize_comm_backend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.layers.attention.backends.paged import tokenspeed_mla
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.utils.env import global_server_args_dict


def run_case(*, rank, degree, context, dtype, queries, draft, block):
    device = torch.device("cuda", rank)
    heads, latent, dim, page, granularity = 16, 512, 576, 64, 128
    blocks = (context + granularity - 1) // granularity + 1
    blocks = (blocks + degree - 1) // degree * degree
    batch = 3
    num_extends = int(queries > 1 and not draft)
    spec_queries = 4 if draft else queries
    spec = MLAConfig(
        backend_name="hybrid_linear_attn",
        num_attention_heads=heads * degree,
        num_kv_heads=1,
        head_dim=dim,
        attn_tp_size=degree,
        kv_lora_rank=latent,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=dim,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=dtype,
        kv_cache_quant_method="",
        prefix_granularity=granularity,
        context_len=blocks * granularity,
        max_bs=batch + num_extends,
        speculative_num_draft_tokens=spec_queries,
        is_draft=draft,
        draft_block_decode=block,
        dcp_size=degree,
        dcp_rank=rank,
        dcp_group=tuple(range(degree)),
        components=(spec,),
    )
    # Only bypass prefill compilation and the production dispatch gate. Decode,
    # metadata kernels and communications below are the actual implementations.
    with patch.object(tokenspeed_mla, "warmup_compile_prefill", lambda **kw: None):
        leaf = tokenspeed_mla.CuteDSLMLABackend(config, spec, kernel_page_size=page)
    leaf.set_cache_pool(object())
    leaf.configure_runtime(
        block_granularity=granularity,
        virtual_block_count=blocks + 1,
        shard_count=degree,
    )
    leaf.init_cuda_graph_state(batch + num_extends)
    torch.manual_seed(173)
    full_cache = torch.randn(
        blocks + 1, granularity, dim, device=device, dtype=torch.bfloat16
    ).to(dtype)
    full_cache[0].zero_()
    local_cache = full_cache[
        [0] + list(range(rank + 1, blocks + 1, degree))
    ].contiguous()
    pool = SimpleNamespace(get_key_buffer=lambda layer_id: local_cache)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=heads,
        head_dim=dim,
        v_head_dim=latent,
        scaling=spec.scaling,
        sliding_window_size=-1,
        k_scale_float=1.0,
    )
    # Nonconsecutive physical blocks: logical page position is not its owner.
    order = torch.cat(
        (
            torch.ones(1, device=device, dtype=torch.int32),
            torch.arange(blocks, 1, -1, device=device, dtype=torch.int32),
        )
    )
    pages = (order[:, None] * 2 + torch.arange(2, device=device)).flatten().int()
    table = pages[None].expand(batch, -1).clone()
    table[-1].zero_()
    ends = torch.tensor([context - 1, 8, 0], device=device, dtype=torch.int32)
    all_q = torch.randn(
        batch, queries, heads * degree, dim, device=device, dtype=torch.bfloat16
    )
    q = all_q[:, :, rank * heads : (rank + 1) * heads].contiguous().flatten(0, 1)

    def refresh():
        # A mixed round places extend rows before the decode slice consumed
        # here; this verifies that DCP uses the same request offset.
        round_ends = torch.cat((ends.new_zeros(num_extends), ends))
        round_table = torch.cat((table.new_zeros((num_extends, table.shape[1])), table))
        leaf.refresh_decode_metadata(
            batch + num_extends,
            batch - 1 + num_extends,
            round_ends,
            round_table,
            num_extends=num_extends,
        )

    def forward():
        return leaf.forward_decode(
            q, None, None, layer, None, pool, batch, save_kv_cache=False
        ).view(batch, queries, heads, latent)

    def reference():
        offsets = (
            torch.zeros(queries, device=device, dtype=torch.int32)
            if block
            else torch.arange(1 - queries, 1, device=device, dtype=torch.int32)
        )
        visible = (ends[:, None] + offsets).clamp_min(0)
        return tokenspeed_mla.tokenspeed_mla_decode(
            query=q.view(batch, queries, heads, dim).to(dtype),
            kv_cache=full_cache.view(-1, page, dim),
            workspace_buffer=leaf._cutedsl_workspace(queries),
            kv_lora_rank=latent,
            qk_rope_head_dim=64,
            block_tables=table,
            seq_lens=ends,
            max_seq_len=blocks * granularity,
            softmax_scale=spec.scaling,
            causal_mask=not block,
            local_visible_lens=visible,
        )

    refresh()
    # Warm up JIT and collective state before capture.
    for _ in range(3):
        actual = forward()
    expected = reference()
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
    assert not actual[-1].any()
    if rank > 0:
        assert (
            leaf.forward_decode_metadata.dcp.local_seq_lens[num_extends + 1].item() == 0
        )
    torch.cuda.synchronize()
    leaf._workspace_pool.freeze()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_output = forward()
    # A later round crosses a scheduler block boundary; graph pointers persist.
    ends[0] = context + 1
    refresh()
    graph.replay()
    expected = reference()
    torch.testing.assert_close(replay_output, expected, atol=0.003, rtol=0.03)
    assert not replay_output[-1].any()
    # Draft updates change visibility without rebuilding the compact table.
    if draft:
        ends[0] = context - 3
        if block:
            leaf.fill_block_decode_seq_lens(batch, ends)
        else:
            leaf.advance_draft_forward_metadata(ends)
        graph.replay()
        expected = reference()
        torch.testing.assert_close(replay_output, expected, atol=0.003, rtol=0.03)
    leaf._workspace_pool.unfreeze()
    if rank == 0:
        print(
            f"PASS dtype={dtype} context={context} Q={queries} draft={draft} "
            f"block={block}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 65536])
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    degree = dist.get_world_size()
    group = tuple(range(degree))
    process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    global_server_args_dict.update(
        mapping=Mapping(rank=rank, world_size=degree),
        chunked_prefill_size=64,
        max_prefill_tokens=64,
        max_model_len=max(args.contexts),
        max_num_seqs=3,
        speculative_algorithm=None,
    )
    initialize_comm_backend(use_pynccl=False)
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for context in args.contexts:
            for queries, draft, block in (
                (1, False, False),
                (4, False, False),
                (1, True, False),
                (4, True, True),
            ):
                run_case(
                    rank=rank,
                    degree=degree,
                    context=context,
                    dtype=dtype,
                    queries=queries,
                    draft=draft,
                    block=block,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
