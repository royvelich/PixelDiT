import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Tuple

from .pixeldit_c2i import PatchTokenEmbedder, PixelTokenEmbedder, PiTBlock
from .modules import (
    FinalLayer,
    FeedForward,
    RMSNorm,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    precompute_freqs_cis_2d,
)


def merge_views(t: torch.Tensor, num_views: int) -> torch.Tensor:
    """``[B*V, H, N, D]`` -> ``[B, H, V*N, D]``: put every view of a sample on one token axis.

    This is what makes cross-view attention possible: after the merge a query from any view of
    sample ``b`` can see the keys of every view of sample ``b``, and of no other sample.

    Correctness rests on ONE convention -- the row axis is **b-major**, i.e. row ``r = b*V + v``,
    which is what ``rearrange('b v c h w -> (b v) c h w')`` produces. Rows ``[b*V .. b*V+V-1]``
    are then sample ``b``'s views, contiguous, so ``view(B, V, ...)`` splits them correctly.
    Building the batch as a ``cat`` of per-view batches instead would be v-major and would
    silently let sample 0 attend to sample 1's reference -- invisible at batch size 1, which is
    what we train at.

    The output token index is ``v*N + n`` (view slowest), so each view occupies one contiguous
    block. ``split_views`` is the exact inverse.
    """
    if num_views == 1:
        return t
    BV, H, N, D = t.shape
    if BV % num_views:
        raise ValueError(f"batch {BV} is not divisible by num_views {num_views}")
    B = BV // num_views
    return t.view(B, num_views, H, N, D).permute(0, 2, 1, 3, 4).reshape(B, H, num_views * N, D)


def split_views(t: torch.Tensor, num_views: int, tokens_per_view: int) -> torch.Tensor:
    """``[B, H, V*N, D]`` -> ``[B*V, H, N, D]``: the exact inverse of ``merge_views``."""
    if num_views == 1:
        return t
    B, H, VN, D = t.shape
    if VN != num_views * tokens_per_view:
        raise ValueError(f"token axis {VN} != num_views {num_views} * {tokens_per_view}")
    return (
        t.view(B, H, num_views, tokens_per_view, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B * num_views, H, tokens_per_view, D)
    )


class PatchGeometryEmbedder(nn.Module):
    """Project per-patch camera/depth features into the image-token space.

    The final projection is zero-initialized, so enabling the module leaves the
    pretrained PixelDiT function exactly unchanged at step zero.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.dropout_probability = float(dropout_probability)
        self.proj_in = nn.Linear(self.input_dim, int(hidden_dim))
        self.proj_out = nn.Linear(int(hidden_dim), int(output_dim))
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, geometry: torch.Tensor, num_views: int) -> torch.Tensor:
        if geometry.dim() != 3 or geometry.shape[-1] != self.input_dim:
            raise ValueError(
                "patch geometry must be [B*V,L,F] with "
                f"F={self.input_dim}, got {tuple(geometry.shape)}"
            )
        if geometry.shape[0] % num_views:
            raise ValueError(
                f"patch geometry rows {geometry.shape[0]} are not divisible by "
                f"num_views={num_views}"
            )
        if self.training and self.dropout_probability > 0:
            samples = geometry.shape[0] // num_views
            keep = (
                torch.rand(
                    samples, 1, 1, 1, device=geometry.device
                )
                >= self.dropout_probability
            )
            geometry = (
                geometry.view(samples, num_views, *geometry.shape[1:])
                * keep.to(geometry.dtype)
            ).reshape_as(geometry)
        return self.proj_out(F.silu(self.proj_in(geometry)))


class GeometryLocalAuxAttention(nn.Module):
    """Independent local cross-attention from render tokens to each auxiliary.

    Reference normalization is deliberately per auxiliary reference.  The
    zero-initialized output projection makes the entire module an exact no-op
    when first attached to a pretrained rank-0 model.
    """

    geometry_dim = 6

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        window_radius: int,
        gate_hidden_dim: int,
        gate_bias: float,
        query_chunk_size: int,
        residual_scale: float,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.window_radius = int(window_radius)
        self.query_chunk_size = int(query_chunk_size)
        self.residual_scale = float(residual_scale)

        self.q_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.dim, bias=True)
        self.confidence = nn.Sequential(
            nn.Linear(8, int(gate_hidden_dim), bias=True),
            nn.SiLU(),
            nn.Linear(int(gate_hidden_dim), 1, bias=True),
        )

        offsets = [
            (dy, dx)
            for dy in range(-self.window_radius, self.window_radius + 1)
            for dx in range(-self.window_radius, self.window_radius + 1)
        ]
        self.register_buffer(
            "local_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )
        self.center_offset_index = offsets.index((0, 0))
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.constant_(self.confidence[-1].bias, float(gate_bias))

    def _project(self, layer, tokens):
        shape = tokens.shape[:-1]
        return layer(tokens).view(
            *shape, self.num_heads, self.head_dim
        )

    def forward(
        self,
        render: torch.Tensor,
        auxiliaries: torch.Tensor,
        geometry: torch.Tensor,
        active: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if render.dim() != 3:
            raise ValueError(
                f"render tokens must be [B,L,C], got {tuple(render.shape)}"
            )
        if auxiliaries.dim() != 4:
            raise ValueError(
                "auxiliary tokens must be [B,A,L,C], got "
                f"{tuple(auxiliaries.shape)}"
            )
        batch, length, channels = render.shape
        if auxiliaries.shape[0] != batch or auxiliaries.shape[2:] != (
            length,
            channels,
        ):
            raise ValueError(
                "render/auxiliary token shapes disagree: "
                f"{tuple(render.shape)} and {tuple(auxiliaries.shape)}"
            )
        auxiliary_count = auxiliaries.shape[1]
        expected_geometry = (
            batch,
            auxiliary_count,
            length,
            self.geometry_dim,
        )
        if tuple(geometry.shape) != expected_geometry:
            raise ValueError(
                f"geometry must be {expected_geometry}, got {tuple(geometry.shape)}"
            )
        if tuple(active.shape) != (batch, auxiliary_count):
            raise ValueError(
                f"active must be {(batch, auxiliary_count)}, got {tuple(active.shape)}"
            )
        if length != int(grid_height) * int(grid_width):
            raise ValueError(
                f"token count {length} != grid {grid_height}x{grid_width}"
            )

        q = self._project(self.q_proj, render).permute(0, 2, 1, 3)
        k = self._project(self.k_proj, auxiliaries).permute(0, 1, 3, 2, 4)
        v = self._project(self.v_proj, auxiliaries).permute(0, 1, 3, 2, 4)
        q = self.q_norm(q)
        k = self.k_norm(k)

        offsets = self.local_offsets.to(device=geometry.device)
        center_x = geometry[..., 0].round().long()
        center_y = geometry[..., 1].round().long()
        local_x = center_x[..., None] + offsets[:, 1]
        local_y = center_y[..., None] + offsets[:, 0]
        inside = (
            (local_x >= 0)
            & (local_x < int(grid_width))
            & (local_y >= 0)
            & (local_y < int(grid_height))
        )
        local_indices = (
            local_y.clamp(0, int(grid_height) - 1) * int(grid_width)
            + local_x.clamp(0, int(grid_width) - 1)
        )
        token_valid = (
            (geometry[..., 3] > 0.5)
            & (active[:, :, None] > 0.5)
        )
        local_valid = inside & token_valid[..., None]
        window_size = offsets.shape[0]

        correction_chunks = []
        corr_sum = render.new_zeros(batch, dtype=torch.float32)
        corr_count = render.new_zeros(batch, dtype=torch.float32)
        gate_sum = render.new_zeros(batch, dtype=torch.float32)
        gate_count = render.new_zeros(batch, dtype=torch.float32)
        scale = self.head_dim ** -0.5

        for start in range(0, length, self.query_chunk_size):
            stop = min(length, start + self.query_chunk_size)
            chunk = stop - start
            indices = local_indices[:, :, start:stop]
            gather_index = indices[:, :, None, :, :, None].expand(
                batch,
                auxiliary_count,
                self.num_heads,
                chunk,
                window_size,
                self.head_dim,
            )
            key_source = k[:, :, :, :, None, :].expand(
                batch,
                auxiliary_count,
                self.num_heads,
                length,
                window_size,
                self.head_dim,
            )
            value_source = v[:, :, :, :, None, :].expand_as(key_source)
            local_k = torch.gather(key_source, 3, gather_index)
            local_v = torch.gather(value_source, 3, gather_index)
            q_chunk = q[:, None, :, start:stop, None, :]
            scores = (q_chunk * local_k).sum(dim=-1) * scale
            valid = local_valid[:, :, None, start:stop]
            scores = scores.masked_fill(
                ~valid,
                -torch.finfo(scores.dtype).max,
            )
            weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            has_candidate = valid.any(dim=-1)
            weights = weights * has_candidate[..., None].to(weights.dtype)

            message = (weights[..., None] * local_v).sum(dim=-2)
            matched_key = (weights[..., None] * local_k).sum(dim=-2)
            entropy = -(
                weights.float()
                * weights.float().clamp_min(1e-8).log()
            ).sum(dim=-1)
            entropy = entropy / max(math.log(float(window_size)), 1.0)
            best = scores.float().amax(dim=-1)
            if window_size > 1:
                top2 = scores.float().topk(2, dim=-1).values
                margin = top2[..., 0] - top2[..., 1]
            else:
                margin = torch.zeros_like(best)
            center_probability = weights[
                ..., self.center_offset_index
            ].float()
            q_for_cos = F.normalize(
                q[:, None, :, start:stop].float(),
                dim=-1,
            )
            key_for_cos = F.normalize(matched_key.float(), dim=-1)
            feature_cosine = (q_for_cos * key_for_cos).sum(dim=-1)

            geom = geometry[:, :, start:stop]
            gate_features = torch.stack(
                [
                    entropy.mean(dim=2),
                    torch.tanh(best.mean(dim=2)),
                    torch.tanh(margin.mean(dim=2)),
                    center_probability.mean(dim=2),
                    geom[..., 2].float(),
                    geom[..., 4].float(),
                    geom[..., 5].float(),
                    feature_cosine.mean(dim=2),
                ],
                dim=-1,
            ).to(render.dtype)
            gate = torch.sigmoid(self.confidence(gate_features))
            valid_token_chunk = token_valid[:, :, start:stop]
            gate = gate * valid_token_chunk[..., None].to(gate.dtype)

            delta = message.permute(0, 1, 3, 2, 4).reshape(
                batch,
                auxiliary_count,
                chunk,
                self.inner_dim,
            )
            numerator = (gate * delta).sum(dim=1)
            denominator = gate.sum(dim=1)
            correction_chunks.append(
                # A confidence sum below one attenuates the whole residual,
                # allowing the router to use neither auxiliary.  Above one,
                # normalize to a confidence-weighted mean so the correction
                # scale does not grow with the reference count.
                numerator / denominator.clamp_min(1.0)
            )

            center_mean = center_probability.mean(dim=2)
            corr = -center_mean.clamp_min(1e-8).log()
            corr_sum += (
                corr * valid_token_chunk.float()
            ).sum(dim=(1, 2))
            corr_count += valid_token_chunk.float().sum(dim=(1, 2))
            gate_sum += (
                gate[..., 0].float() * valid_token_chunk.float()
            ).sum(dim=(1, 2))
            gate_count += valid_token_chunk.float().sum(dim=(1, 2))

        combined = torch.cat(correction_chunks, dim=1)
        projected = self.out_proj(combined) * self.residual_scale
        correction_rms = projected.float().square().mean(
            dim=(1, 2)
        ).sqrt()
        stats = {
            "correspondence_loss": corr_sum / corr_count.clamp_min(1.0),
            "gate_mean": gate_sum / gate_count.clamp_min(1.0),
            "valid_fraction": token_valid.float().mean(dim=(1, 2)),
            "correction_rms": correction_rms,
        }
        return projected, stats


class MMDiTJointAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            pos_img: torch.Tensor,
            pos_txt: torch.Tensor = None,
            attn_mask: torch.Tensor = None,
            num_views: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # y is None when the text stream is disabled: the attention then runs single-stream over
        # the image tokens alone. Everything text-shaped below is skipped rather than fed zeros, so
        # no text weight is touched and none of its compute is paid.
        #
        # With text present: image tokens arrive at B*V (one row per view) while the TEXT stream
        # stays at B, shared across a sample's views. Merging text as well would duplicate it V
        # times and let the copies diverge, so only the image slice joins the cross-view token axis.
        B, Nx, C = x.shape
        text_on = y is not None
        By = B // num_views
        if text_on:
            By, Ny, Cy = y.shape
            assert C == Cy, "x and y must share the channel dim"
            assert B == By * num_views, (
                f"image batch {B} must be text batch {By} * num_views {num_views}"
            )

        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        # Note the batch: image tokens are at B (== B*V), text at By. Using B here would be a
        # shape error at V>1 and a silent no-op at V=1.
        if text_on:
            qkv_y = self.qkv_y(y).reshape(By, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
            qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
            qy = self.q_norm_y(qy)
            ky = self.k_norm_y(ky)

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if text_on and pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)

        if text_on:
            qy = qy.transpose(1, 2)
            ky = ky.transpose(1, 2)
            vy = vy.transpose(1, 2)

        # Merge AFTER RoPE: every view shares the same H_s x W_s grid, so each view must get
        # the same positional phase. Merging first would be identical here, but doing it last
        # keeps the invariant obvious.
        qx = merge_views(qx, num_views)
        kx = merge_views(kx, num_views)
        vx = merge_views(vx, num_views)

        if text_on:
            q_joint = torch.cat([qy, qx], dim=2)
            k_joint = torch.cat([ky, kx], dim=2)
            v_joint = torch.cat([vy, vx], dim=2)
        else:
            q_joint, k_joint, v_joint = qx, kx, vx

        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        n_txt = Ny if text_on else 0
        out_x = split_views(out_joint[:, :, n_txt:, :], num_views, Nx)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)
        out_x = self.proj_drop_x(self.proj_x(out_x))

        if not text_on:
            return out_x, None
        out_y = out_joint[:, :, :Ny, :].transpose(1, 2).reshape(By, Ny, C)
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation_img=None, adaLN_modulation_txt=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation_img = adaLN_modulation_img if adaLN_modulation_img is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = adaLN_modulation_txt if adaLN_modulation_txt is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None, num_views=1):
        # `c` is per IMAGE row, i.e. at B*V. The text stream is at B, so its modulation must be
        # fed the render rows only: b-major rows mean sample b's render is row b*V, which is
        # exactly `c[::num_views]`. A no-op at num_views=1.
        text_on = y is not None
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        if text_on:
            c_txt = c if num_views == 1 else c[::num_views]
            shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c_txt).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y) if text_on else None
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask, num_views)
        x = x + gate_msa_x * attn_x
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        if not text_on:
            return x, None

        y = y + gate_msa_y * attn_y
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class PixDiT_T2I(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = -1,
        use_pixel_abs_pos: bool = True,
        pit_adaln_post_modulation: bool = False,
        patch_geometry_enabled: bool = False,
        patch_geometry_input_dim: int = 11,
        patch_geometry_hidden_dim: int = 256,
        patch_geometry_dropout: float = 0.0,
        depth_channel_enabled: bool = False,
        depth_channel_dim: int = 1,
        geometry_local_aux_enabled: bool = False,
        geometry_local_aux_blocks=(),
        geometry_local_aux_window_radius: int = 2,
        geometry_local_aux_heads: int = 8,
        geometry_local_aux_head_dim: int = 64,
        geometry_local_aux_query_chunk_size: int = 256,
        geometry_local_aux_gate_hidden_dim: int = 128,
        geometry_local_aux_gate_bias: float = -2.0,
        geometry_local_aux_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_encoder_index = int(repa_encoder_index)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.patch_geometry_enabled = bool(patch_geometry_enabled)
        self.depth_channel_enabled = bool(depth_channel_enabled)
        self.depth_channel_dim = int(depth_channel_dim)
        self.geometry_local_aux_enabled = bool(geometry_local_aux_enabled)
        if self.depth_channel_enabled and self.depth_channel_dim < 1:
            raise ValueError("depth_channel_dim must be >= 1 when enabled")
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")

        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size ** 2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))

        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList([
            MMDiTBlockT2I(
                self.hidden_size,
                self.num_groups,
                adaLN_modulation_img=self._shared_cond_adaln_img,
                adaLN_modulation_txt=self._shared_cond_adaln_txt,
            )
            for _ in range(self.patch_depth)
        ])
        self.geometry_local_aux_attentions = nn.ModuleDict()
        if self.geometry_local_aux_enabled:
            block_indices = [int(index) for index in geometry_local_aux_blocks]
            invalid = [
                index
                for index in block_indices
                if index < 0 or index >= self.patch_depth
            ]
            if invalid:
                raise ValueError(
                    "geometry-local auxiliary block indices exceed patch depth "
                    f"{self.patch_depth}: {invalid}"
                )
            for index in block_indices:
                self.geometry_local_aux_attentions[str(index)] = (
                    GeometryLocalAuxAttention(
                        self.hidden_size,
                        geometry_local_aux_heads,
                        geometry_local_aux_head_dim,
                        geometry_local_aux_window_radius,
                        geometry_local_aux_gate_hidden_dim,
                        geometry_local_aux_gate_bias,
                        geometry_local_aux_query_chunk_size,
                        geometry_local_aux_residual_scale,
                    )
                )
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                    rope_fn=precompute_freqs_cis_2d,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )

        self.patch_geometry_embedder = None
        if self.patch_geometry_enabled:
            self.patch_geometry_embedder = PatchGeometryEmbedder(
                patch_geometry_input_dim,
                patch_geometry_hidden_dim,
                self.hidden_size,
                patch_geometry_dropout,
            )

        # Algebraically this is a projection of concatenated RGBD, split into the pretrained RGB
        # weights plus two new depth-only terms. Splitting it preserves every RGB checkpoint
        # tensor and lets depth start as an exact no-op.
        self.patch_depth_embedder = None
        self.pixel_depth_embedder = None
        if self.depth_channel_enabled:
            self.patch_depth_embedder = nn.Linear(
                self.depth_channel_dim * self.patch_size * self.patch_size,
                self.hidden_size,
                bias=False,
            )
            self.pixel_depth_embedder = nn.Linear(
                self.depth_channel_dim,
                self.pixel_hidden_size,
                bias=False,
            )

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
        self.last_repa_tokens = None
        self.last_geometry_local_aux_stats = None

        self.initialize_weights()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.patch_depth_embedder is not None:
            nn.init.zeros_(self.patch_depth_embedder.weight)
            nn.init.zeros_(self.pixel_depth_embedder.weight)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _prepare_depth(
        self,
        depth: torch.Tensor,
        model_rows: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if depth is None:
            if self.depth_channel_enabled:
                raise ValueError(
                    "depth_channel_enabled=True requires depth "
                    f"[B*V,{self.depth_channel_dim},H,W]"
                )
            return None
        if not self.depth_channel_enabled:
            raise ValueError(
                "depth was supplied but depth_channel_enabled=False"
            )
        if depth.dim() != 4 or depth.shape[1] != self.depth_channel_dim:
            raise ValueError(
                "depth must be "
                f"[B*V,{self.depth_channel_dim},H,W], got {tuple(depth.shape)}"
            )
        if tuple(depth.shape[-2:]) != (height, width):
            raise ValueError(
                f"depth spatial shape {tuple(depth.shape[-2:])} != {(height, width)}"
            )
        if model_rows % depth.shape[0]:
            raise ValueError(
                f"depth rows {depth.shape[0]} do not divide model rows {model_rows}"
            )
        if depth.shape[0] != model_rows:
            depth = depth.repeat(model_rows // depth.shape[0], 1, 1, 1)
        return depth.to(device=device, dtype=dtype)

    def _embed_pixel_depth(
        self,
        depth: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        B = depth.shape[0]
        Hs, Ws = height // self.patch_size, width // self.patch_size
        projected = self.pixel_depth_embedder(
            depth.permute(0, 2, 3, 1).contiguous()
        )
        projected = projected.view(
            B,
            Hs,
            self.patch_size,
            Ws,
            self.patch_size,
            self.pixel_hidden_size,
        )
        projected = projected.permute(0, 1, 3, 2, 4, 5).contiguous()
        return projected.view(
            B * Hs * Ws,
            self.patch_size * self.patch_size,
            self.pixel_hidden_size,
        )

    @staticmethod
    def _prepare_patch_geometry(
        geometry: torch.Tensor,
        model_rows: int,
        num_views: int,
        token_count: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if geometry.dim() == 4:
            if geometry.shape[1] != num_views:
                raise ValueError(
                    f"patch geometry has {geometry.shape[1]} views, "
                    f"expected {num_views}"
                )
            geometry = geometry.reshape(
                geometry.shape[0] * geometry.shape[1],
                geometry.shape[2],
                geometry.shape[3],
            )
        if geometry.dim() != 3:
            raise ValueError(
                f"patch geometry must be [B,V,L,F] or [B*V,L,F], "
                f"got {tuple(geometry.shape)}"
            )
        if geometry.shape[1] != token_count:
            raise ValueError(
                f"patch geometry token count {geometry.shape[1]} != {token_count}"
            )
        if model_rows % geometry.shape[0]:
            raise ValueError(
                f"patch geometry rows {geometry.shape[0]} do not divide "
                f"model rows {model_rows}"
            )
        if geometry.shape[0] != model_rows:
            geometry = geometry.repeat(
                model_rows // geometry.shape[0], 1, 1
            )
        return geometry.to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_geometry_local_aux(
        geometry: torch.Tensor,
        active: torch.Tensor,
        model_rows: int,
        num_views: int,
        token_count: int,
        device,
        dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_rows = model_rows // num_views
        auxiliary_count = num_views - 2
        if geometry.dim() != 4:
            raise ValueError(
                "geometry-local auxiliary data must be [B,A,L,F], got "
                f"{tuple(geometry.shape)}"
            )
        if geometry.shape[1:] != (
            auxiliary_count,
            token_count,
            GeometryLocalAuxAttention.geometry_dim,
        ):
            raise ValueError(
                "geometry-local auxiliary shape disagrees with model views: "
                f"got {tuple(geometry.shape)}, expected [B,{auxiliary_count},"
                f"{token_count},{GeometryLocalAuxAttention.geometry_dim}]"
            )
        if active.dim() != 2 or active.shape[1] != auxiliary_count:
            raise ValueError(
                f"geometry-local active mask must be [B,{auxiliary_count}], "
                f"got {tuple(active.shape)}"
            )
        if geometry.shape[0] != active.shape[0]:
            raise ValueError(
                "geometry and active-mask batch sizes disagree: "
                f"{geometry.shape[0]} and {active.shape[0]}"
            )
        if sample_rows % geometry.shape[0]:
            raise ValueError(
                f"geometry batch {geometry.shape[0]} does not divide "
                f"model sample batch {sample_rows}"
            )
        repeats = sample_rows // geometry.shape[0]
        if repeats != 1:
            geometry = geometry.repeat(repeats, 1, 1, 1)
            active = active.repeat(repeats, 1)
        return (
            geometry.to(device=device, dtype=dtype),
            active.to(device=device, dtype=dtype),
        )

    def forward(
        self,
        x,
        t,
        y=None,
        s=None,
        mask=None,
        num_views=1,
        patch_geometry=None,
        depth=None,
        geometry_local_aux=None,
        geometry_local_aux_active=None,
        geometry_local_aux_disable=False,
    ):
        """``num_views`` > 1 means ``x`` holds V views per sample, stacked **b-major** on the
        batch axis (row ``b*V + v``); see ``merge_views``. The views meet only inside the patch
        blocks' joint attention. Everything else -- the pixel blocks, ``final_layer``, the
        returned shape -- treats them as independent rows, so the output stays ``[B*V,3,H,W]``
        and the caller reshapes if it wants a view axis. ``num_views=1`` is a bit-exact no-op.

        ``t`` is per row (length ``B*V``); ``y`` and ``mask`` are per SAMPLE (length ``B``),
        because the text stream is shared across a sample's views.
        """
        B, _, H, W = x.shape          # B here is B*V: one row per (sample, view)
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws
        depth = self._prepare_depth(
            depth,
            B,
            H,
            W,
            x.device,
            x.dtype,
        )
        self.last_geometry_local_aux_stats = None
        if self.geometry_local_aux_enabled:
            if num_views < 3:
                raise ValueError(
                    "geometry-local auxiliary fusion needs num_views >= 3 "
                    "(render, rank-0, and at least one auxiliary)"
                )
            if geometry_local_aux is None or geometry_local_aux_active is None:
                raise ValueError(
                    "geometry-local auxiliary fusion requires geometry and "
                    "an active-reference mask"
                )
            geometry_local_aux, geometry_local_aux_active = (
                self._prepare_geometry_local_aux(
                    geometry_local_aux,
                    geometry_local_aux_active,
                    B,
                    num_views,
                    L,
                    x.device,
                    x.dtype,
                )
            )
        elif (
            geometry_local_aux is not None
            or geometry_local_aux_active is not None
            or geometry_local_aux_disable
        ):
            raise ValueError(
                "geometry-local auxiliary inputs were supplied while the "
                "model mode is disabled"
            )

        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        text_on = y is not None
        y_emb, Ltxt, Btxt = None, 0, B // num_views
        if text_on:
            if y.dim() != 3:
                raise ValueError("Text embedding y must be [B, L, D]")
            Ltxt = min(y.shape[1], self.txt_max_length)
            y = y[:, :Ltxt, :]
            Btxt = y.shape[0]                 # == B // num_views
            y_emb = self.y_embedder(y).view(Btxt, Ltxt, self.hidden_size)
            y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)

        condition = torch.nn.functional.silu(t_emb)

        if s is None:
            s0 = self.s_embedder(x_patches)
            if depth is not None:
                depth_patches = F.unfold(
                    depth,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ).transpose(1, 2)
                s0 = s0 + self.patch_depth_embedder(depth_patches)
            if self.patch_geometry_embedder is not None and patch_geometry is not None:
                geometry = self._prepare_patch_geometry(
                    patch_geometry,
                    B,
                    num_views,
                    L,
                    x.device,
                    s0.dtype,
                )
                s0 = s0 + self.patch_geometry_embedder(geometry, num_views)
            pos_txt = (self.fetch_pos_text(Ltxt, x.device)
                       if (text_on and self.use_text_rope) else None)
            attn_mask_joint = None
            # The mask exists only to pad the TEXT axis; the image half is always all-False, so
            # with no text there is nothing to mask and the whole tensor is skipped.
            if text_on and mask is not None and isinstance(mask, torch.Tensor):
                m = mask
                while m.dim() > 2 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 3 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 2:
                    pad = (m == 0)
                    # The key axis is [text | V*L image tokens] at batch Btxt after the merge.
                    # Image tokens are never padded, so their half stays all-False.
                    pad_img = torch.zeros((Btxt, num_views * L), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(
                        Btxt, 1, 1, Ltxt + num_views * L
                    )
            self.last_repa_tokens = None
            s = s0
            if not self.geometry_local_aux_enabled:
                for i in range(self.patch_depth):
                    s, y_emb = self.patch_blocks[i](
                        s,
                        y_emb,
                        condition,
                        pos,
                        pos_txt,
                        attn_mask_joint,
                        num_views,
                    )
                    if 0 < self.repa_encoder_index == (i + 1):
                        self.last_repa_tokens = s
            else:
                sample_batch = B // num_views
                auxiliary_count = num_views - 2
                block_stats = []
                if text_on and attn_mask_joint is not None:
                    base_mask = torch.cat(
                        [
                            attn_mask_joint[..., :Ltxt],
                            torch.zeros(
                                sample_batch,
                                1,
                                1,
                                2 * L,
                                dtype=torch.bool,
                                device=x.device,
                            ),
                        ],
                        dim=-1,
                    )
                else:
                    base_mask = None
                for i in range(self.patch_depth):
                    views = s.view(
                        sample_batch,
                        num_views,
                        L,
                        self.hidden_size,
                    )
                    cond_views = condition.view(
                        sample_batch,
                        num_views,
                        1,
                        self.hidden_size,
                    )
                    base_rows = views[:, :2].reshape(
                        sample_batch * 2,
                        L,
                        self.hidden_size,
                    )
                    base_cond = cond_views[:, :2].reshape(
                        sample_batch * 2,
                        1,
                        self.hidden_size,
                    )
                    base_rows, y_emb = self.patch_blocks[i](
                        base_rows,
                        y_emb,
                        base_cond,
                        pos,
                        pos_txt,
                        base_mask,
                        2,
                    )
                    auxiliary_rows = views[:, 2:].reshape(
                        sample_batch * auxiliary_count,
                        L,
                        self.hidden_size,
                    )
                    auxiliary_cond = cond_views[:, 2:].reshape(
                        sample_batch * auxiliary_count,
                        1,
                        self.hidden_size,
                    )
                    auxiliary_rows, _ = self.patch_blocks[i](
                        auxiliary_rows,
                        None,
                        auxiliary_cond,
                        pos,
                        None,
                        None,
                        1,
                    )
                    views = torch.cat(
                        [
                            base_rows.view(
                                sample_batch,
                                2,
                                L,
                                self.hidden_size,
                            ),
                            auxiliary_rows.view(
                                sample_batch,
                                auxiliary_count,
                                L,
                                self.hidden_size,
                            ),
                        ],
                        dim=1,
                    )
                    attention = (
                        self.geometry_local_aux_attentions[str(i)]
                        if str(i) in self.geometry_local_aux_attentions
                        else None
                    )
                    if attention is not None and not geometry_local_aux_disable:
                        correction, stats = attention(
                            views[:, 0],
                            views[:, 2:],
                            geometry_local_aux,
                            geometry_local_aux_active,
                            Hs,
                            Ws,
                        )
                        render = views[:, 0] + correction
                        views = torch.cat(
                            [render[:, None], views[:, 1:]],
                            dim=1,
                        )
                        block_stats.append(stats)
                    s = views.reshape(B, L, self.hidden_size)
                    if 0 < self.repa_encoder_index == (i + 1):
                        self.last_repa_tokens = s
                if block_stats:
                    self.last_geometry_local_aux_stats = {
                        name: torch.stack(
                            [stats[name] for stats in block_stats],
                            dim=0,
                        ).mean(dim=0)
                        for name in block_stats[0]
                    }
            s = torch.nn.functional.silu(t_emb + s)
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        batch_size, length, _ = s.shape
        if length != L:
            if length > L:
                s = s[:, :L, :]
            else:
                pad_len = L - length
                s = torch.cat([s, s.new_zeros(B, pad_len, s.shape[2])], dim=1)
            length = L

        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        if depth is not None:
            x_pixels = x_pixels + self._embed_pixel_depth(depth, H, W)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
