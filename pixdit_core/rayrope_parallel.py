"""Parallel RayRoPE cross-attention for a frozen PixelDiT DiT trunk.

The RayRoPE geometry/attention implementation is imported from the authors'
official repository, pinned as ``PixelDiT/third_party/RayRoPE``.  This file is
only an adapter between PixelDiT's token/camera layouts and that implementation;
it deliberately does not reimplement the RayRoPE transforms.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from pathlib import Path
import sys
from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .modules import RMSNorm


def _install_official_cross_attention_kv_count_fix(module) -> None:
    """Correct the released cross-attention's one-query/K-reference reshape.

    The official helper receives a camera count explicitly. Its released
    cross-attention caller passes the query-camera count for both query and KV
    geometry, so one render query plus K references fails whenever K != 1.
    Inferring the count from the official position tensors preserves the exact
    official helper and every RayRoPE transform while making the documented
    cross-attention API work for unequal query/KV camera counts.
    """
    current = module._prepare_rope_coeff_uniformd
    if getattr(current, "_pixeldit_kv_count_fix", False):
        return

    @wraps(current)
    def prepare_with_tensor_camera_count(
        positions,
        num_freqs,
        freq_base,
        batch,
        _camera_count,
        num_patches,
    ):
        first_position = next(iter(positions.values()))
        camera_count = first_position.shape[1]
        return current(
            positions,
            num_freqs,
            freq_base,
            batch,
            camera_count,
            num_patches,
        )

    prepare_with_tensor_camera_count._pixeldit_kv_count_fix = True
    module._prepare_rope_coeff_uniformd = prepare_with_tensor_camera_count


@lru_cache(maxsize=1)
def _official_rayrope_symbols():
    root = Path(__file__).resolve().parents[1] / "third_party" / "RayRoPE"
    if not (root / "pos_enc" / "rayrope_cross_attention.py").is_file():
        raise ImportError(
            "the pinned official RayRoPE submodule is missing; run "
            "`git submodule update --init --recursive` from the repository root"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from pos_enc import rayrope_cross_attention  # type: ignore
    from pos_enc.utils.functional import (  # type: ignore
        camera_to_raymap,
        patchify,
        raymap_to_plucker,
    )

    _install_official_cross_attention_kv_count_fix(
        rayrope_cross_attention
    )
    return (
        rayrope_cross_attention.RayRoPE_DotProductAttention_Cross,
        camera_to_raymap,
        patchify,
        raymap_to_plucker,
    )


def normalized_to_pixel_intrinsics(
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Undo the corpus' centered-normalized intrinsic convention."""
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in [3,3]")
    output = torch.zeros_like(intrinsics, dtype=torch.float32)
    source = intrinsics.float()
    output[..., 0, 0] = source[..., 0, 0] * float(width)
    output[..., 1, 1] = source[..., 1, 1] * float(height)
    output[..., 0, 2] = (source[..., 0, 2] + 0.5) * float(width)
    output[..., 1, 2] = (source[..., 1, 2] + 0.5) * float(height)
    output[..., 2, 2] = 1.0
    return output


def build_official_camray_patches(
    normalized_intrinsics: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the authors' intrinsics-only CamRay/Pluecker patch features.

    Their ``camray`` mode calls ``camera_to_raymap`` with identity extrinsics,
    converts the result to Pluecker coordinates, and patchifies it.
    """
    _, camera_to_raymap, patchify, raymap_to_plucker = (
        _official_rayrope_symbols()
    )
    pixel_intrinsics = normalized_to_pixel_intrinsics(
        normalized_intrinsics, height, width
    )
    identities = torch.eye(
        4,
        device=pixel_intrinsics.device,
        dtype=pixel_intrinsics.dtype,
    ).expand(pixel_intrinsics.shape[:-2] + (4, 4))
    raymap = camera_to_raymap(
        Ks=pixel_intrinsics,
        camtoworlds=identities,
        height=height,
        width=width,
    )
    patches = patchify(raymap_to_plucker(raymap), patch_size)
    batch, views = patches.shape[:2]
    return patches.reshape(batch * views, *patches.shape[2:]).to(dtype=dtype)


class ParallelRayRoPECrossAttention(nn.Module):
    """Trainable render-query/reference-KV adapter using official RayRoPE.

    PixelDiT's original attention is external to this module and can remain
    fully frozen.  This adapter updates only the render row; reference rows get
    a zero residual.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        num_heads: int = 8,
        head_dim: int = 96,
        ray_token_dim: int = 0,
        pos_enc_type: str = "d_pj+0_3d",
        num_rays_per_patch: int = 3,
        depth_type: str = "predict_dsig",
        denc_type: str = "d",
        freq_base: float = 3.0,
        apply_vo: bool = True,
        init_depth: float = 0.0,
        init_sigma: float = 3.0,
        residual_scale: float = 0.1,
        learnable_scale: bool = False,
        mask_invalid_depth: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.width = self.num_heads * self.head_dim
        self.ray_token_dim = int(ray_token_dim)
        self.pos_enc_type = str(pos_enc_type)
        self.num_rays_per_patch = int(num_rays_per_patch)
        self.depth_type = str(depth_type)
        self.denc_type = str(denc_type)
        self.freq_base = float(freq_base)
        self.apply_vo = bool(apply_vo)
        self.mask_invalid_depth = bool(mask_invalid_depth)

        if self.num_heads < 1:
            raise ValueError("parallel RayRoPE num_heads must be positive")
        if self.head_dim < 1 or self.head_dim % 24:
            raise ValueError(
                "parallel RayRoPE head_dim must be a positive multiple of 24"
            )
        if self.pos_enc_type != "d_pj+0_3d":
            raise ValueError(
                "official parallel RayRoPE requires pos_enc_type='d_pj+0_3d'"
            )
        if self.num_rays_per_patch != 3:
            raise ValueError(
                "official parallel RayRoPE requires three rays per patch"
            )
        if self.depth_type not in ("predict_dsig", "cached_gs"):
            raise ValueError(
                "parallel RayRoPE depth_type must be predict_dsig|cached_gs"
            )
        if self.denc_type != "d":
            raise ValueError("official parallel RayRoPE requires denc_type='d'")
        if not self.apply_vo:
            raise ValueError(
                "official parallel RayRoPE requires Q/K/V/output transforms"
            )

        self.input_norm = RMSNorm(self.hidden_size, eps=1e-6)
        self.token_projection = nn.Linear(
            self.hidden_size, self.width, bias=False
        )
        self.ray_token_projection = (
            nn.Linear(self.ray_token_dim, self.width, bias=False)
            if self.ray_token_dim > 0
            else None
        )
        self.q_projection = nn.Linear(self.width, self.width, bias=False)
        self.k_projection = nn.Linear(self.width, self.width, bias=False)
        self.v_projection = nn.Linear(self.width, self.width, bias=False)
        self.output_projection = nn.Linear(
            self.width, self.hidden_size, bias=False
        )
        self.depth_projection = None
        if self.depth_type == "predict_dsig":
            self.depth_projection = nn.Linear(self.width, 2, bias=True)
            nn.init.zeros_(self.depth_projection.weight)
            with torch.no_grad():
                self.depth_projection.bias.copy_(
                    torch.tensor(
                        [float(init_depth), float(init_sigma)],
                        dtype=self.depth_projection.bias.dtype,
                    )
                )

        scale = torch.tensor(float(residual_scale))
        if learnable_scale:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale, persistent=True)
        self.learnable_scale = bool(learnable_scale)
        self._official_attention_cache = {}

    def _official_attention(
        self,
        patch_height: int,
        patch_width: int,
        image_height: int,
        image_width: int,
    ):
        key = (
            int(patch_height),
            int(patch_width),
            int(image_height),
            int(image_width),
        )
        attention = self._official_attention_cache.get(key)
        if attention is None:
            RayRoPECross, _, _, _ = _official_rayrope_symbols()
            attention = RayRoPECross(
                head_dim=self.head_dim,
                patches_x=patch_width,
                patches_y=patch_height,
                image_width=image_width,
                image_height=image_height,
                pos_enc_type=self.pos_enc_type,
                num_rays_per_patch=self.num_rays_per_patch,
                depth_type="predict_dsig",
                denc_type=self.denc_type,
                freq_base=self.freq_base,
                apply_vo=self.apply_vo,
            )
            # The released method is decorated with torch.compile at import
            # time, before the unequal-Q/KV-camera compatibility shim above
            # can replace its global coefficient helper. Bind the unchanged
            # underlying official method so it resolves the corrected helper
            # at runtime. The attention and RayRoPE math remain official.
            compiled_prepare = getattr(
                type(attention), "_prepare_apply_fns", None
            )
            raw_prepare = getattr(compiled_prepare, "__wrapped__", None)
            if raw_prepare is not None:
                attention._prepare_apply_fns = MethodType(
                    raw_prepare, attention
                )
            self._official_attention_cache[key] = attention
        return attention

    def _project_heads(
        self, projection: nn.Linear, tokens: torch.Tensor
    ) -> torch.Tensor:
        batch, length = tokens.shape[:2]
        return (
            projection(tokens)
            .view(batch, length, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _depth_inputs(
        self,
        projected_views: torch.Tensor,
        depth: Optional[torch.Tensor],
        validity: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch, views, tokens = projected_views.shape[:3]
        if self.depth_type == "predict_dsig":
            if depth is not None or validity is not None:
                raise ValueError(
                    "cached depth was supplied to predict_dsig parallel RayRoPE"
                )
            predicted = self.depth_projection(projected_views)
            valid = torch.ones(
                batch,
                views,
                tokens,
                device=projected_views.device,
                dtype=torch.bool,
            )
        else:
            expected = (batch, views, tokens)
            if depth is None or tuple(depth.shape) != expected:
                raise ValueError(
                    f"parallel cached depth must be {expected}, got "
                    f"{None if depth is None else tuple(depth.shape)}"
                )
            if validity is None or tuple(validity.shape) != expected:
                raise ValueError(
                    f"parallel cached validity must be {expected}, got "
                    f"{None if validity is None else tuple(validity.shape)}"
                )
            valid = validity.to(
                device=projected_views.device, dtype=torch.bool
            )
            safe_depth = depth.to(
                device=projected_views.device, dtype=torch.float32
            ).clamp_min(1e-2)
            predicted = torch.stack(
                [torch.log(safe_depth), torch.zeros_like(safe_depth)], dim=-1
            )

        query_depth = predicted[:, 0]
        reference_depth = predicted[:, 1:].reshape(
            batch, (views - 1) * tokens, 2
        )
        query_valid = valid[:, 0]
        reference_valid = valid[:, 1:].reshape(
            batch, (views - 1) * tokens
        )

        attention_mask = None
        reference_available = reference_valid.any(dim=-1)
        if self.mask_invalid_depth and not bool(reference_valid.all()):
            # SDPA cannot consume an all-masked row. Temporarily expose all
            # keys for those samples and zero their final residual below.
            safe_valid = reference_valid.clone()
            safe_valid[~reference_available] = True
            attention_mask = torch.zeros(
                batch,
                1,
                1,
                safe_valid.shape[-1],
                device=projected_views.device,
                dtype=projected_views.dtype,
            )
            attention_mask.masked_fill_(
                ~safe_valid[:, None, None], float("-inf")
            )
        output_valid = query_valid & reference_available[:, None]
        return query_depth, reference_depth, attention_mask, output_valid

    def forward(
        self,
        x: torch.Tensor,
        num_views: int,
        viewmats: torch.Tensor,
        normalized_intrinsics: torch.Tensor,
        patch_grid: Tuple[int, int],
        image_size: Tuple[int, int],
        ray_token_patches: Optional[torch.Tensor] = None,
        camera_keep: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        depth_validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if num_views <= 1:
            return torch.zeros_like(x)
        rows, tokens, channels = x.shape
        if channels != self.hidden_size or rows % num_views:
            raise ValueError(
                "parallel RayRoPE received incompatible token dimensions"
            )
        batch = rows // num_views
        patch_height, patch_width = (int(v) for v in patch_grid)
        image_height, image_width = (int(v) for v in image_size)
        if patch_height * patch_width != tokens:
            raise ValueError("parallel RayRoPE patch grid/token mismatch")
        if (
            image_height != patch_height * self.patch_size
            or image_width != patch_width * self.patch_size
        ):
            raise ValueError(
                "parallel RayRoPE image size, patch grid and patch size "
                "are inconsistent"
            )
        if viewmats.shape != (batch, num_views, 4, 4):
            raise ValueError("parallel RayRoPE view matrices have wrong shape")
        if normalized_intrinsics.shape != (batch, num_views, 3, 3):
            raise ValueError("parallel RayRoPE intrinsics have wrong shape")

        normalized = self.input_norm(x)
        projected = self.token_projection(normalized)
        if self.ray_token_projection is not None:
            if ray_token_patches is None:
                raise ValueError(
                    "parallel RayRoPE ray-token encoding requires patches"
                )
            expected = (rows, tokens, self.ray_token_dim)
            if tuple(ray_token_patches.shape) != expected:
                raise ValueError(
                    f"ray token patches must be {expected}, got "
                    f"{tuple(ray_token_patches.shape)}"
                )
            projected = projected + self.ray_token_projection(
                ray_token_patches.to(projected.dtype)
            )
        elif ray_token_patches is not None:
            raise ValueError(
                "ray token patches supplied while token encoding is disabled"
            )

        projected_views = projected.view(
            batch, num_views, tokens, self.width
        )
        query_tokens = projected_views[:, 0]
        reference_tokens = projected_views[:, 1:].reshape(
            batch, (num_views - 1) * tokens, self.width
        )
        query = self._project_heads(self.q_projection, query_tokens)
        key = self._project_heads(self.k_projection, reference_tokens)
        value = self._project_heads(self.v_projection, reference_tokens)

        query_depth, reference_depth, attention_mask, output_valid = (
            self._depth_inputs(
                projected_views, depth, depth_validity
            )
        )
        pixel_intrinsics = normalized_to_pixel_intrinsics(
            normalized_intrinsics, image_height, image_width
        )
        official_attention = self._official_attention(
            patch_height,
            patch_width,
            image_height,
            image_width,
        )
        official_attention._precompute_and_cache_apply_fns(
            w2cs=viewmats[:, :1].float(),
            Ks=pixel_intrinsics[:, :1].float(),
            w2cs_kv=viewmats[:, 1:].float(),
            Ks_kv=pixel_intrinsics[:, 1:].float(),
            context_depths=None,
        )
        output = official_attention(
            query,
            key,
            value,
            predicted_d=query_depth.float(),
            predicted_d_kv=reference_depth.float(),
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        output = (
            output.permute(0, 2, 1, 3)
            .reshape(batch, tokens, self.width)
            .to(projected.dtype)
        )
        output = self.output_projection(output)
        output = output * output_valid[..., None].to(output.dtype)
        if camera_keep is not None:
            if camera_keep.shape != (batch, 1, 1):
                raise ValueError("camera_keep has wrong shape")
            output = output * camera_keep.to(output.dtype)
        output = output * self.residual_scale.to(output.dtype)
        reference_zeros = output.new_zeros(
            batch, num_views - 1, tokens, channels
        )
        return torch.cat(
            [output[:, None], reference_zeros], dim=1
        ).reshape_as(x)
