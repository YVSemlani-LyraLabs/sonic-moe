# Acknowledgement: this file is adapted from
# https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py

# /*
# * Copyright 2018-2020 Philippe Tillet
# * Copyright 2020-2022 OpenAI
# *
# * Permission is hereby granted, free of charge, to any person obtaining
# * a copy of this software and associated documentation files
# * (the "Software"), to deal in the Software without restriction,
# * including without limitation the rights to use, copy, modify, merge,
# * publish, distribute, sublicense, and/or sell copies of the Software,
# * and to permit persons to whom the Software is furnished to do so,
# * subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be
# * included in all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# */

import torch
import triton
import triton.language as tl

from sonicmoe.count_cumsum import count_cumsum


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L33
@triton.jit
def _keyed_add(x, y):
    # Segmented add for associative_scan: upper 16 bits = expert id (key),
    # lower 16 bits = within-expert count. If keys match, accumulate count;
    # otherwise reset to y (start of new segment).
    key_mask: tl.constexpr = 0xFFFF0000
    kx = x & key_mask
    ky = y & key_mask
    return tl.where(kx == ky, x + y - kx, y)


@triton.jit
def _compute_col_partial_sum_kernel(
    topk_indices_ptr,
    partial_sum_ptr,
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,  # next_power_of_2(K),
    K: tl.constexpr,  # actual number of experts per token
    E_POW2: tl.constexpr,  # next_power_of_2(E)
):
    # One CTA per tile. Tile `t` covers tokens [t * TOKENS_PER_TILE, (t+1) * TOKENS_PER_TILE).
    # Produces partial_sum[e, tile_id] = number of entries in this tile routed to expert e.
    # Layout: partial_sum is [E, n_tiles] (row-major), so partial_sum[e, t] = partial_sum_ptr + e * n_tiles + t.
    # Caller transposes to [n_tiles, E] before passing to stage1/stage2.
    tile_id = tl.program_id(0)

    # Zero this tile's column in partial_sum[*, tile_id].
    # Chunked by E_POW2 to keep vector width a power of 2.
    for e_start in tl.static_range(0, E, E_POW2):
        e_offs = e_start + tl.arange(0, E_POW2)
        tl.store(
            partial_sum_ptr + e_offs * n_tiles + tile_id,
            tl.zeros([E_POW2], tl.int32),
            mask=e_offs < E,
        )

    # Load expert ids for this tile: shape [TOKENS_PER_TILE, K_POW2].
    # Tokens beyond T and k-slots beyond K are masked out (other=-1).
    tok_offs = tile_id * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    k_offs = tl.arange(0, K_POW2)
    tok_mask = tok_offs < T

    load_mask = tok_mask[:, None] & (k_offs[None, :] < K)
    safe_k = tl.minimum(k_offs, K - 1)  # avoid OOB when k_offs >= K
    expert_ids = tl.load(
        topk_indices_ptr + tok_offs[:, None] * K + safe_k[None, :],
        mask=load_mask,
        other=-1,
    )

    # Flatten to [TOKENS_PER_TILE * K_POW2] and histogram into partial_sum.
    # safe_experts remaps masked (-1) entries to expert 0 (harmless: flat_mask=False).
    flat_experts = tl.reshape(expert_ids, [TOKENS_PER_TILE * K_POW2])
    flat_mask = tl.reshape(load_mask, [TOKENS_PER_TILE * K_POW2])
    safe_experts = tl.where(flat_mask, flat_experts, 0)

    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([TOKENS_PER_TILE * K_POW2], 1, dtype=tl.int32),
        mask=flat_mask,
    )


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L44
@triton.jit
def _bitmatrix_metadata_compute_stage1(
    expert_freq_offset_ptr,
    expert_offs_ptr,
    E: tl.constexpr,
    partial_sum_ptr,
    n_tiles,
    BLOCK_M: tl.constexpr,  # chunk size for iterating over tiles per expert
    BLOCK_N: tl.constexpr,  # chunk size for iterating over experts in cumsum
):
    # Assume grid size == E + 1

    pid = tl.program_id(0)
    if pid < E:
        # convert partial_sum[e, *] from raw counts to exclusive prefix
        # sums over tiles. After this kernel, partial_sum[e, t] =
        # number of entries for expert e in tiles 0..t-1.

        # This is read by stage2 to locate each entry's position within expert e's contiguous output segment.
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0)
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        # Exclusive prefix sum of per-expert total counts → expert_offs[e].
        # expert_freq_offset[e] = total entries routed to expert e (from A.sum(dim=1)).
        # expert_offs[e] = sum of expert_freq_offset[0..e-1] = global start of expert e.
        curr_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            freq = tl.load(expert_freq_offset_ptr + offs, mask=offs < E, other=0)
            excl_cumsum = tl.cumsum(freq, 0) - freq + curr_sum
            curr_sum += tl.sum(freq, 0)
            tl.store(expert_offs_ptr + offs, excl_cumsum, mask=offs < E)


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L44
@triton.jit
def _bitmatrix_metadata_compute_stage2(
    s_scatter_idx_ptr,
    s_reverse_scatter_idx_ptr,
    x_gather_idx_ptr,
    topk_indices_ptr,
    T,
    partial_sum_ptr,
    n_tiles,
    expert_offs_ptr,
    K_POW2: tl.constexpr,  # padded K, == BLOCK_SIZE / BLOCK
    K: tl.constexpr,  # actual experts per token
    TOKENS_PER_BLOCK: tl.constexpr,  # tokens per tile
):
    # One CTA per tile, same tiling as _compute_col_partial_sum_kernel.
    # For each entry (token t, k-slot k) in this tile:
    #   s_reverse_scatter_idx[entry_idx] = output position in expert-sorted order
    #   s_scatter_idx[output_pos]        = entry_idx   (inverse permutation)
    #   x_gather_idx[output_pos]         = token index (= entry_idx // K)
    #
    # Output position = expert_offs[e]          (global start of expert e)
    #                 + partial_sum[tile, e]     (entries for e in earlier tiles, after stage1)
    #                 + within_expert_rank       (position within this tile's group for e)
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2  # fast path: no padding waste
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE)  # position within this tile's flat [BLOCK*K_POW2] space
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < T * K_POW2

    # Load expert id for each slot. IS_POW2_K fast path reads topk_indices as a
    # flat 1D array (no padding gaps). Non-pow2 path reads 2D with k_slot masking.
    if IS_POW2_K:
        expert = tl.load(topk_indices_ptr + offs_global, mask=mask, other=-1).to(tl.uint32)
    else:
        token_i_local = offs_local // K_POW2
        k_slot = offs_local % K_POW2
        token_i_global = pid_m * TOKENS_PER_BLOCK + token_i_local
        load_mask = mask & (k_slot < K)
        safe_k = tl.minimum(k_slot, K - 1)
        expert = tl.load(
            topk_indices_ptr + token_i_global * K + safe_k,
            mask=load_mask,
            other=-1,
        ).to(tl.uint32)

    # Pack (expert, presort_offs) into a uint32 kv pair and sort by expert.
    # Upper 16 bits = expert id (sort key), lower 16 bits = pre-sort local offset.
    # Invalid slots have expert=0xffff (from other=-1 cast to uint32 >> 16).
    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF  # exclude padding/OOB slots

    # Segmented scan to compute within-expert rank (0-based exclusive count).
    # scan_input packs expert id in upper 16 bits and count=1 in lower 16 bits.
    # _keyed_add resets the count at each expert boundary.
    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, _keyed_add)
    within_expert_rank = (inclusive_run_lengths - 1) & 0xFFFF  # exclusive = inclusive - 1

    # Output position for this entry in the expert-sorted output array.
    # partial_sum layout after stage1: [n_tiles, E], stride (1, n_tiles) since it is A.T.
    # So partial_sum[pid_m, expert] = partial_sum_ptr + pid_m*1 + expert*n_tiles.
    s_reverse_scatter_idx = tl.load(partial_sum_ptr + pid_m + expert * n_tiles, mask=mask)
    s_reverse_scatter_idx += tl.load(expert_offs_ptr + expert, mask=mask)
    s_reverse_scatter_idx += within_expert_rank

    if IS_POW2_K:
        # presort_offs == offs_local before sort; entry_idx is the flat index into
        # topk_router_indices.view(-1), i.e. token * K + k_slot.
        presort_offs = kv_pairs & 0xFFFF
        entry_idx = pid_m * BLOCK_SIZE + presort_offs
        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask)
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse_scatter_idx, entry_idx // K_POW2, mask=mask)
    else:
        # presort_offs is in K_POW2-padded space; convert to unpadded entry_idx.
        presort_offs = kv_pairs & 0xFFFF
        token_i_global_s = pid_m * TOKENS_PER_BLOCK + presort_offs // K_POW2
        entry_idx = token_i_global_s * K + presort_offs % K_POW2
        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask)
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse_scatter_idx, token_i_global_s, mask=mask)


def TC_topk_router_metadata_triton(
    topk_router_indices: torch.Tensor, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T, K = topk_router_indices.size()
    TK = T * K
    device = topk_router_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(32, min(256, 1024 // K_POW2))
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    # ── Kernel 1: tiled histogram ─────────────────────────────────────────────
    # col_partial_sum_trans[E, n_tiles]: raw per-expert-per-tile counts.
    # Stored transposed so each CTA writes to its own column (tile_id), avoiding
    # cross-CTA write conflicts. Transposed back to [n_tiles, E] for stage1/stage2.
    col_partial_sum_trans = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _compute_col_partial_sum_kernel[(n_tiles,)](
        topk_router_indices,
        col_partial_sum_trans,
        T,
        E,
        n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )
    col_partial_sum = col_partial_sum_trans.T  # [n_tiles, E]

    # ── Kernel 2: stage1 ─────────────────────────────────────────────────────
    # - For each expert e (pid < E): convert col_partial_sum[*, e] from raw
    #   counts to exclusive prefix sums over tiles in-place.
    # - For pid == E: write exclusive cumsum of expert_freq_offset into
    #   expert_freq_off[0:E] (= col_offs, a view into expert_freq_off).
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = col_partial_sum_trans.sum(dim=1)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    col_offs = expert_frequency_offset[:E]

    STAGE1_BLOCK_M = max(32, min(512, triton.next_power_of_2(n_tiles)))
    STAGE1_BLOCK_N = max(32, min(512, E_POW2))
    _bitmatrix_metadata_compute_stage1[(E + 1,)](
        expert_frequency,
        col_offs,
        E,
        col_partial_sum,
        n_tiles,
        BLOCK_M=STAGE1_BLOCK_M,
        BLOCK_N=STAGE1_BLOCK_N,
    )

    # ── Kernel 3: stage2 ─────────────────────────────────────────────────────
    # For each tile: sort entries by expert, compute output positions, scatter.
    _bitmatrix_metadata_compute_stage2[(n_tiles,)](
        s_scatter_idx,
        s_reverse_scatter_idx,
        x_gather_idx,
        topk_router_indices,
        T,
        col_partial_sum,
        n_tiles,
        col_offs,
        K_POW2=K_POW2,
        TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
        K=K,
        num_warps=4,
        num_stages=2,
    )

    expert_frequency_offset[E] = TK

    return (expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx)


def TC_topk_router_metadata_torch(
    topk_router_indices: torch.Tensor, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    K = topk_router_indices.size(1)
    s_scatter_idx = torch.argsort(topk_router_indices.view(-1)).int()
    expert_frequency, expert_frequency_offset = count_cumsum(topk_router_indices.view(-1), E, do_cumsum=True)
    expert_frequency_offset = torch.cat(
        [
            torch.zeros(1, device=expert_frequency_offset.device, dtype=expert_frequency_offset.dtype),
            expert_frequency_offset,
        ]
    )
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.size(0), device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    x_gather_idx = s_scatter_idx // K

    return (expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx)


def TC_topk_router_metadata(
    topk_router_indices: torch.Tensor, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return TC_topk_router_metadata_triton(topk_router_indices, E)
