# MIT License
#
# Copyright (c) Authors of "Cameras as Relative Positional Encoding"
# https://arxiv.org/abs/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from NVIDIA ArtiFixer's PRoPE implementation for PixelDiT joint
# image attention. The reference implementation is Apache-2.0/MIT licensed.

"""Camera-relative positional encoding for dense multi-view attention.

PRoPE transforms Q, K, V, and the attention output using each view's calibrated
camera. The token axis must be laid out as contiguous view blocks:
``[view0 patches | view1 patches | ...]``, which is exactly PixelDiT's
``merge_views`` convention.
"""

from functools import partial
from typing import Callable, List, Tuple

import torch
from torch import nn


def invert_se3(matrix: torch.Tensor) -> torch.Tensor:
    """Invert batched rigid 4x4 transforms without a generic matrix inverse."""
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"SE3 matrices must end in [4,4], got {tuple(matrix.shape)}")
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3]
    rotation_inv = rotation.transpose(-1, -2)
    output = torch.zeros_like(matrix)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation_inv, translation
    )
    output[..., 3, 3] = 1
    return output


class PropeDotProductAttention(nn.Module):
    """PRoPE transforms for one dot-product attention operation."""

    def __init__(
        self,
        head_dim: int,
        patches_x: int = 0,
        patches_y: int = 0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"PRoPE head_dim must be divisible by 4, got {head_dim}")
        if freq_base <= 0 or freq_scale <= 0:
            raise ValueError("PRoPE frequency base and scale must be positive")
        self.head_dim = int(head_dim)
        self.patches_x = int(patches_x)
        self.patches_y = int(patches_y)
        self.freq_base = float(freq_base)
        self.freq_scale = float(freq_scale)
        self.cameras = 0
        self.apply_fn_q = None
        self.apply_fn_kv = None
        self.apply_fn_o = None
        self.register_buffer("coeffs_x_0", torch.empty(0), persistent=False)
        self.register_buffer("coeffs_x_1", torch.empty(0), persistent=False)
        self.register_buffer("coeffs_y_0", torch.empty(0), persistent=False)
        self.register_buffer("coeffs_y_1", torch.empty(0), persistent=False)

    def update_coeffs(
        self,
        patches_x: int,
        patches_y: int,
        device: torch.device,
    ) -> None:
        if (
            self.patches_x == patches_x
            and self.patches_y == patches_y
            and self.coeffs_x_0.numel()
            and self.coeffs_x_0.device == device
        ):
            return
        self.patches_x = int(patches_x)
        self.patches_y = int(patches_y)
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y,)),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        coeffs_y = _rope_precompute_coeffs(
            torch.repeat_interleave(
                torch.arange(patches_y, device=device), patches_x
            ),
            freq_base=self.freq_base,
            freq_scale=self.freq_scale,
            feat_dim=self.head_dim // 4,
        )
        self.coeffs_x_0, self.coeffs_x_1 = coeffs_x
        self.coeffs_y_0, self.coeffs_y_1 = coeffs_y

    def prepare(self, viewmats: torch.Tensor, intrinsics: torch.Tensor) -> None:
        batch, cameras, _, _ = viewmats.shape
        if intrinsics.shape != (batch, cameras, 3, 3):
            raise ValueError(
                f"PRoPE intrinsics must be {(batch, cameras, 3, 3)}, "
                f"got {tuple(intrinsics.shape)}"
            )
        self.cameras = cameras
        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            intrinsics=intrinsics,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def apply_to_q(self, query: torch.Tensor) -> torch.Tensor:
        self._check_features(query, "query")
        return self.apply_fn_q(query)

    def apply_to_kv(self, features: torch.Tensor) -> torch.Tensor:
        self._check_features(features, "key/value")
        return self.apply_fn_kv(features)

    def apply_to_output(self, output: torch.Tensor) -> torch.Tensor:
        self._check_features(output, "output")
        return self.apply_fn_o(output)

    def _check_features(self, features: torch.Tensor, name: str) -> None:
        if self.apply_fn_q is None:
            raise RuntimeError("PRoPE camera transforms were not prepared")
        expected_tokens = self.cameras * self.patches_x * self.patches_y
        if features.dim() != 4:
            raise ValueError(f"PRoPE {name} must be [B,H,N,D]")
        if features.shape[2] != expected_tokens:
            raise ValueError(
                f"PRoPE {name} has {features.shape[2]} tokens, "
                f"expected {expected_tokens}"
            )
        if features.shape[-1] != self.head_dim:
            raise ValueError(
                f"PRoPE {name} head width {features.shape[-1]} != {self.head_dim}"
            )


def _prepare_apply_fns(
    head_dim: int,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    coeffs_x: Tuple[torch.Tensor, torch.Tensor],
    coeffs_y: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    projection = torch.einsum(
        "...ij,...jk->...ik", _lift_intrinsics(intrinsics), viewmats
    )
    projection_t = projection.transpose(-1, -2)
    projection_inv = torch.einsum(
        "...ij,...jk->...ik",
        invert_se3(viewmats),
        _lift_intrinsics(_invert_intrinsics(intrinsics)),
    )

    transforms_q = [
        (partial(_apply_tiled_matrix, matrix=projection_t), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_matrix, matrix=projection_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_output = [
        (partial(_apply_tiled_matrix, matrix=projection), head_dim // 2),
        (
            partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True),
            head_dim // 4,
        ),
        (
            partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True),
            head_dim // 4,
        ),
    ]
    return (
        partial(_apply_block_diagonal, func_size_pairs=transforms_q),
        partial(_apply_block_diagonal, func_size_pairs=transforms_kv),
        partial(_apply_block_diagonal, func_size_pairs=transforms_output),
    )


def _apply_tiled_matrix(
    features: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    batch, heads, sequence, feature_dim = features.shape
    cameras = matrix.shape[1]
    matrix_dim = matrix.shape[-1]
    if matrix.shape != (batch, cameras, matrix_dim, matrix_dim):
        raise ValueError("invalid PRoPE camera matrix shape")
    if sequence % cameras or feature_dim % matrix_dim:
        raise ValueError("PRoPE features do not divide into camera/matrix blocks")
    reshaped = features.reshape(
        batch,
        heads,
        cameras,
        sequence // cameras,
        feature_dim // matrix_dim,
        matrix_dim,
    )
    return torch.einsum(
        "bcij,bncpkj->bncpki", matrix, reshaped
    ).reshape_as(features)


def _rope_precompute_coeffs(
    positions: torch.Tensor,
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if feat_dim % 2:
        raise ValueError(f"PRoPE positional feature width must be even, got {feat_dim}")
    num_freqs = feat_dim // 2
    frequencies = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * frequencies
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    features: torch.Tensor,
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    cosine, sine = coeffs
    if cosine.shape[2] != features.shape[2]:
        repeats = features.shape[2] // cosine.shape[2]
        cosine = cosine.repeat(1, 1, repeats, 1)
        sine = sine.repeat(1, 1, repeats, 1)
    if cosine.shape[-1] != features.shape[-1] // 2:
        raise ValueError("PRoPE coefficient and feature widths disagree")
    x_part, y_part = features.chunk(2, dim=-1)
    if inverse:
        return torch.cat(
            [
                cosine * x_part - sine * y_part,
                sine * x_part + cosine * y_part,
            ],
            dim=-1,
        )
    return torch.cat(
        [
            cosine * x_part + sine * y_part,
            -sine * x_part + cosine * y_part,
        ],
        dim=-1,
    )


def _apply_block_diagonal(
    features: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    functions, block_sizes = zip(*func_size_pairs)
    if features.shape[-1] != sum(block_sizes):
        raise ValueError("PRoPE block sizes do not cover the attention head")
    output = torch.empty_like(features)
    start = 0
    for function, block, block_size in zip(
        functions, torch.split(features, block_sizes, dim=-1), block_sizes
    ):
        end = start + block_size
        output[..., start:end] = function(block)
        start = end
    return output


def _lift_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("camera intrinsics must end in [3,3]")
    output = torch.zeros(
        intrinsics.shape[:-2] + (4, 4),
        dtype=intrinsics.dtype,
        device=intrinsics.device,
    )
    output[..., :3, :3] = intrinsics
    output[..., 3, 3] = 1
    return output


def _invert_intrinsics(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("camera intrinsics must end in [3,3]")
    output = torch.zeros_like(intrinsics)
    output[..., 0, 0] = 1 / intrinsics[..., 0, 0]
    output[..., 1, 1] = 1 / intrinsics[..., 1, 1]
    output[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    output[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    output[..., 2, 2] = 1
    return output
