"""RayRoPE geometry and rotary transforms for multi-view patch attention.

This file adapts the official RayRoPE implementation released by Yu Wu et al.:
https://github.com/yu-wu/rayrope, commit
4495152eb8bbb27a34db1259212574d46a365cef.

The mathematical path intentionally follows the paper implementation:

* ``d_pj+0_3d`` positions with three rays per patch;
* one predicted log-depth and uncertainty per visual token;
* uniform-depth-interval expected sine/cosine coefficients;
* query-view-relative transforms for every key/value view;
* rotary transforms on Q, K, V, followed by the inverse output transform.

The integration API is stateless so activation checkpoint recomputation cannot
observe mutable camera state. PixelDiT already supplies normalized intrinsics,
so this adapter starts immediately after the official ``normalize_K`` step.

MIT License

Copyright (c) 2026 Yu Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import torch


MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
MAX_D_F = 10.0


@dataclass(frozen=True)
class RayRoPECoefficients:
    """Rotary coefficients for one multi-view attention invocation."""

    query_cos: torch.Tensor
    query_sin: torch.Tensor
    key_value_cos: Tuple[torch.Tensor, ...]
    key_value_sin: Tuple[torch.Tensor, ...]


class RayRoPEEncoding(torch.nn.Module):
    """Official-paper RayRoPE encoding, exposed as a stateless coefficient API."""

    def __init__(
        self,
        head_dim: int,
        pos_enc_type: str = "d_pj+0_3d",
        num_rays_per_patch: int = 3,
        depth_type: str = "predict_dsig",
        denc_type: str = "d",
        freq_base: float = 3.0,
        apply_vo: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.pos_enc_type = str(pos_enc_type)
        self.num_rays_per_patch = int(num_rays_per_patch)
        self.depth_type = str(depth_type)
        self.denc_type = str(denc_type)
        self.freq_base = float(freq_base)
        self.apply_vo = bool(apply_vo)

        # The hybrid PixelDiT mode deliberately implements the paper's released
        # configuration. Keeping this strict prevents a run name from saying
        # "RayRoPE" while silently selecting an unverified ablation.
        if self.pos_enc_type != "d_pj+0_3d":
            raise ValueError(
                "official-paper RayRoPE requires pos_enc_type='d_pj+0_3d'"
            )
        if self.num_rays_per_patch != 3:
            raise ValueError(
                "official-paper RayRoPE requires three rays per patch "
                "(num_rays_per_patch=3)"
            )
        if self.depth_type not in ("predict_dsig", "cached_gs"):
            raise ValueError(
                "RayRoPE depth_type must be 'predict_dsig' or 'cached_gs'"
            )
        if self.denc_type != "d":
            raise ValueError(
                "official-paper RayRoPE requires denc_type='d'"
            )
        if self.freq_base <= 0:
            raise ValueError("RayRoPE freq_base must be positive")
        if not self.apply_vo:
            raise ValueError(
                "official-paper RayRoPE requires Q/K/V and output transforms"
            )

        self.use_p0, self.p0_type = True, "3d"
        self.use_pd, self.pd_type = True, "pj"
        self.use_pinf, self.pinf_type = False, "none"
        self.rope_coord_dim = (
            3 * int(self.use_p0)
            + self.num_rays_per_patch
            * 3
            * (int(self.use_pd) + int(self.use_pinf))
        )
        self.rope_mat_dim = 2 * self.rope_coord_dim
        if self.head_dim % self.rope_mat_dim:
            raise ValueError(
                f"RayRoPE head_dim={self.head_dim} must be a multiple of "
                f"{self.rope_mat_dim} for d_pj+0_3d with three rays"
            )
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim
        self.offsets = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    def prepare_coefficients(
        self,
        w2cs: torch.Tensor,
        normalized_intrinsics: torch.Tensor,
        predicted_depth_uncertainty: torch.Tensor,
        patch_grid: Tuple[int, int],
    ) -> RayRoPECoefficients:
        """Build the exact official Q/KV coefficients for all query views.

        Args:
            w2cs: Camera-from-world matrices ``[B,V,4,4]``.
            normalized_intrinsics: Intrinsics ``[B,V,3,3]`` in centered,
                normalized image coordinates.
            predicted_depth_uncertainty: ``[B,V*N,2]`` containing predicted
                log-depth and uncertainty.
            patch_grid: ``(patches_y, patches_x)``.
        """
        if w2cs.ndim != 4 or w2cs.shape[-2:] != (4, 4):
            raise ValueError("RayRoPE w2cs must be [B,V,4,4]")
        if normalized_intrinsics.shape != w2cs.shape[:2] + (3, 3):
            raise ValueError("RayRoPE intrinsics must be [B,V,3,3]")
        if len(patch_grid) != 2:
            raise ValueError("RayRoPE patch_grid must be (patches_y, patches_x)")
        patches_y, patches_x = (int(value) for value in patch_grid)
        if patches_y <= 0 or patches_x <= 0:
            raise ValueError("RayRoPE patch dimensions must be positive")

        batch, num_cameras = w2cs.shape[:2]
        num_patches = patches_x * patches_y
        if predicted_depth_uncertainty.shape != (
            batch,
            num_cameras * num_patches,
            2,
        ):
            raise ValueError(
                "RayRoPE predicted depth must be "
                f"{(batch, num_cameras * num_patches, 2)}, got "
                f"{tuple(predicted_depth_uncertainty.shape)}"
            )

        # Camera/depth geometry is intentionally evaluated in FP32, as in the
        # authors' implementation. The caller casts encoded Q/K/V back to the
        # attention dtype after applying these coefficients.
        with torch.autocast(device_type=w2cs.device.type, enabled=False):
            w2cs_f = w2cs.float()
            intrinsics_f = normalized_intrinsics.float()
            predicted_f = predicted_depth_uncertainty.float()
            c2ws = _invert_se3(w2cs_f)
            lifted_k = _lift_k(intrinsics_f)
            projection = torch.einsum("...ij,...jk->...ik", lifted_k, w2cs_f)
            projection_inv = torch.einsum(
                "...ij,...jk->...ik",
                c2ws,
                _lift_k(_invert_k(intrinsics_f)),
            )

            p0_world = _get_camera_centers(c2ws, num_patches).unsqueeze(-2)
            depths = _prepare_depths(
                predicted_f,
                batch=batch,
                num_cameras=num_cameras,
                num_patches=num_patches,
                num_rays_per_patch=self.num_rays_per_patch,
            )
            pd_world = _get_point_coords(
                projection_inv,
                patches_x,
                patches_y,
                self.offsets,
                depths,
            )

            positions_query: Dict[str, List[torch.Tensor]] = defaultdict(list)
            kv_cos: List[torch.Tensor] = []
            kv_sin: List[torch.Tensor] = []

            for camera_index in range(num_cameras):
                positions_kv: Dict[str, torch.Tensor] = {}
                projection_query = projection[:, camera_index]
                w2c_query = w2cs_f[:, camera_index]

                p0_3d = _transform_to_query_frame(
                    p0_world,
                    projection_query,
                    w2c_query,
                    transform_type="3d",
                    denc_type=self.denc_type,
                ).flatten(-2, -1)
                positions_kv["p0"] = p0_3d
                positions_query["p0"].append(p0_3d[:, camera_index])

                pd_direction, pd_depth = _transform_to_query_frame(
                    pd_world,
                    projection_query,
                    w2c_query,
                    transform_type="pj",
                    denc_type=self.denc_type,
                )
                pd_direction = (
                    pd_direction.flatten(0, 1)[..., :2].flatten(-2, -1)
                )
                pd_depth = pd_depth.flatten(0, 1).flatten(-2, -1)
                positions_kv["pd_dir"] = pd_direction
                positions_kv["pd_depth"] = pd_depth
                positions_query["pd_dir"].append(
                    pd_direction[:, camera_index]
                )
                positions_query["pd_depth"].append(
                    pd_depth[:, camera_index]
                )

                cosine, sine = _prepare_rope_coeff_uniform_depth(
                    positions_kv,
                    self.num_rope_freqs,
                    self.freq_base,
                    batch,
                    num_cameras,
                    num_patches,
                )
                kv_cos.append(cosine)
                kv_sin.append(sine)

            stacked_query = {
                name: torch.stack(values, dim=1)
                for name, values in positions_query.items()
            }
            query_cos, query_sin = _prepare_rope_coeff_uniform_depth(
                stacked_query,
                self.num_rope_freqs,
                self.freq_base,
                batch,
                num_cameras,
                num_patches,
            )

        return RayRoPECoefficients(
            query_cos=query_cos,
            query_sin=query_sin,
            key_value_cos=tuple(kv_cos),
            key_value_sin=tuple(kv_sin),
        )


def apply_rayrope(
    features: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply the released RayRoPE half-split rotary transform."""
    batch, _, sequence, feature_dim = features.shape
    expected = (batch, sequence, feature_dim // 2)
    if cosine.shape != expected or sine.shape != expected:
        raise ValueError(
            f"RayRoPE coefficients must be {expected}, got "
            f"{tuple(cosine.shape)} and {tuple(sine.shape)}"
        )
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    first, second = torch.chunk(features, 2, dim=-1)
    if inverse:
        rotated_first = first * cosine + second * sine
        rotated_second = -first * sine + second * cosine
    else:
        rotated_first = first * cosine - second * sine
        rotated_second = first * sine + second * cosine
    return torch.cat([rotated_first, rotated_second], dim=-1)


def _transform_to_query_frame(
    points_world: torch.Tensor,
    projection: torch.Tensor,
    w2c: torch.Tensor,
    transform_type: str,
    denc_type: str,
):
    if transform_type == "3d":
        points_camera = torch.einsum(
            "bij,...bcprj->...bcpri", w2c, points_world
        )
        points_camera = points_camera / torch.clamp(
            points_camera[..., -1:], min=1e-4
        )
        return points_camera[..., :3]
    if transform_type != "pj":
        raise ValueError(f"unknown RayRoPE transform type {transform_type!r}")

    points_camera = torch.einsum(
        "bij,...bcprj->...bcpri", projection, points_world
    )
    safe_abs = torch.sqrt(points_camera[..., 2:3].pow(2) + 1e-9)
    z_value = torch.clamp(safe_abs, min=1e-4)
    w_value = torch.clamp(points_camera[..., -1:], min=1e-4)
    direction = points_camera[..., :3] / (
        torch.norm(points_camera[..., :3], dim=-1, keepdim=True) + 1e-6
    )
    if denc_type == "d":
        return direction, z_value / w_value
    raise ValueError(f"unsupported RayRoPE depth encoding {denc_type!r}")


def _get_camera_centers(
    c2ws: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    batch, num_cameras = c2ws.shape[:2]
    centers = c2ws[:, :, :, 3]
    return centers.view(batch, num_cameras, 1, 4).expand(
        batch, num_cameras, num_patches, 4
    )


def _get_point_coords(
    projection_inv: torch.Tensor,
    patches_x: int,
    patches_y: int,
    offsets: Tuple[Tuple[float, float], ...],
    depths: torch.Tensor,
) -> torch.Tensor:
    device = projection_inv.device
    dtype = projection_inv.dtype
    batch, num_cameras = projection_inv.shape[:2]
    num_patches = patches_x * patches_y
    num_rays = len(offsets)

    u_base, v_base = torch.meshgrid(
        torch.arange(patches_x, device=device, dtype=dtype),
        torch.arange(patches_y, device=device, dtype=dtype),
        indexing="xy",
    )
    coordinates = []
    for offset_x, offset_y in offsets:
        u_coord = (u_base + offset_x) / patches_x - 0.5
        v_coord = (v_base + offset_y) / patches_y - 0.5
        coordinates.append(
            torch.stack([u_coord, v_coord], dim=-1).reshape(-1, 2)
        )
    coordinates = torch.stack(coordinates, dim=1)
    coordinates = coordinates.view(
        1, 1, num_patches, num_rays, 2
    ).expand(batch, num_cameras, -1, -1, -1)

    depths = torch.clamp(depths, min=1e-2, max=MAX_DEPTH)
    disparity = 1.0 / depths
    coordinates = coordinates.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
    homogeneous = torch.cat(
        [coordinates, torch.ones_like(disparity), disparity], dim=-1
    )
    return torch.einsum(
        "bcij,ebcprj->ebcpri", projection_inv, homogeneous
    )


def _prepare_depths(
    predicted: torch.Tensor,
    *,
    batch: int,
    num_cameras: int,
    num_patches: int,
    num_rays_per_patch: int,
) -> torch.Tensor:
    predicted = predicted.reshape(
        batch, num_cameras, num_patches, 1, 2
    )
    predicted_log_depth = predicted[..., 0:1]
    predicted_sigma = predicted[..., 1:2]
    depth_min = torch.exp(
        torch.clamp(
            predicted_log_depth - predicted_sigma,
            max=MAX_LOG_DEPTH,
        )
    )
    depth_max = torch.exp(
        torch.clamp(
            predicted_log_depth + predicted_sigma,
            max=MAX_LOG_DEPTH,
        )
    )
    depths = torch.stack([depth_min, depth_max], dim=0)
    depths = depths.expand(
        -1, -1, -1, -1, num_rays_per_patch, -1
    )
    if not torch.isfinite(depths).all():
        raise ValueError("non-finite RayRoPE predicted depths")
    return depths


def _prepare_rope_coeff_uniform_depth(
    positions: Dict[str, torch.Tensor],
    num_freqs: int,
    freq_base: float,
    batch: int,
    num_cameras: int,
    num_patches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coordinate_dim = sum(value.shape[-1] for value in positions.values())
    cosine_parts = []
    sine_parts = []

    for position_name, position in positions.items():
        if position_name in ("p0", "pd_3d"):
            max_period = 4.0
        elif position_name in ("pinf_dir", "pd_dir", "p0_dir"):
            max_period = 8.0
        elif position_name in ("pd_depth", "p0_depth"):
            max_period = MAX_D_F * 2.0 * 4.0
            position = torch.clamp(position, min=-MAX_D_F, max=MAX_D_F)
        else:
            raise ValueError(f"unknown RayRoPE position {position_name!r}")

        min_period = max_period / (freq_base ** (num_freqs - 1))
        frequencies = _get_frequency(
            num_freqs,
            max_period=max_period,
            min_period=min_period,
            device=position.device,
            dtype=position.dtype,
        )
        angles = torch.einsum("f,bcpd->bcpfd", frequencies, position)

        if angles.shape[0] == batch * 2:
            angles_min = angles[:batch]
            angles_max = angles[batch:]
            same = torch.isclose(angles_min, angles_max, atol=1e-2, rtol=0)
            cosine_min = torch.cos(angles_min)
            sine_min = torch.sin(angles_min)
            cosine_max = torch.cos(angles_max)
            sine_max = torch.sin(angles_max)
            delta = angles_max - angles_min
            safe_delta = torch.where(same, torch.ones_like(delta), delta)
            expected_cosine = (sine_max - sine_min) / safe_delta
            expected_sine = (cosine_min - cosine_max) / safe_delta
            cosine = torch.where(same, cosine_min, expected_cosine)
            sine = torch.where(same, sine_min, expected_sine)
        elif angles.shape[0] == batch:
            cosine = torch.cos(angles)
            sine = torch.sin(angles)
        else:
            raise ValueError(
                f"unexpected RayRoPE coefficient batch {angles.shape[0]}"
            )
        cosine_parts.append(cosine)
        sine_parts.append(sine)

    cosine_out = torch.cat(cosine_parts, dim=-1).reshape(
        batch,
        num_cameras * num_patches,
        num_freqs * coordinate_dim,
    )
    sine_out = torch.cat(sine_parts, dim=-1).reshape(
        batch,
        num_cameras * num_patches,
        num_freqs * coordinate_dim,
    )
    return cosine_out.contiguous(), sine_out.contiguous()


def _get_frequency(
    num_freqs: int,
    *,
    max_period: float,
    min_period: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    log_min_frequency = math.log(2.0 * math.pi / max_period)
    log_max_frequency = math.log(2.0 * math.pi / min_period)
    return torch.exp(
        torch.linspace(
            log_min_frequency,
            log_max_frequency,
            num_freqs,
            device=device,
            dtype=dtype,
        )
    )


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inverse = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inverse
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inverse,
        transforms[..., :3, 3],
    )
    output[..., 3, 3] = 1.0
    return output


def _lift_k(intrinsics: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(
        intrinsics.shape[:-2] + (4, 4),
        device=intrinsics.device,
        dtype=intrinsics.dtype,
    )
    output[..., :3, :3] = intrinsics
    output[..., 3, 3] = 1.0
    return output


def _invert_k(intrinsics: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(intrinsics)
    output[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    output[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    output[..., 0, 2] = (
        -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    )
    output[..., 1, 2] = (
        -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    )
    output[..., 2, 2] = 1.0
    return output
