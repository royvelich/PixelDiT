import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from typing import Tuple

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


def _relative_freqs_cis_2d(
    positions: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Per-token 2D rotary phases for arbitrary relative patch positions.

    ``positions`` is ``[...,N,2]`` ordered as ``(x,y)``. Unlike
    ``precompute_freqs_cis_2d``, this supports a different correspondence
    centre for every sample and target patch.
    """
    if head_dim % 4:
        raise ValueError(
            f"2D RoPE requires head_dim divisible by 4, got {head_dim}"
        )
    positions = positions.float()
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(
                0, head_dim, 4, device=positions.device, dtype=torch.float32
            )
            / head_dim
        )
    )
    x_angles = positions[..., 0, None] * freqs
    y_angles = positions[..., 1, None] * freqs
    x_cis = torch.polar(torch.ones_like(x_angles), x_angles)
    y_cis = torch.polar(torch.ones_like(y_angles), y_angles)
    return torch.cat([x_cis, y_cis], dim=-1)


def _apply_token_rotary(
    tensor: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply per-group, per-token rotary phases.

    ``tensor`` is ``[G,N,H,D]`` and ``freqs_cis`` is ``[G,N,D/2]``.
    """
    tensor_complex = torch.view_as_complex(
        tensor.float().reshape(*tensor.shape[:-1], -1, 2)
    )
    output = tensor_complex * freqs_cis[:, :, None, :]
    return torch.view_as_real(output).flatten(-2).type_as(tensor)


class LocalReferenceFusionAttention(nn.Module):
    """Image-only local attention used before PixelDiT's global patch blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.qkv_local = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj_local = nn.Linear(self.dim, self.dim)
        self.attn_drop = float(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        query_count: int,
        positions: torch.Tensor,
        key_valid: torch.Tensor,
        query_valid: torch.Tensor,
        confidence: torch.Tensor,
        pair_geometry: str,
        confidence_bias: float,
        fallback_key_index: int,
    ) -> torch.Tensor:
        groups, token_count, channels = tokens.shape
        qkv = self.qkv_local(tokens).reshape(
            groups,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        q = self.q_norm(qkv[:, :query_count, 0])
        k = self.k_norm(qkv[:, :, 1])
        v = qkv[:, :, 2]

        if pair_geometry == "offset_valid":
            phases = _relative_freqs_cis_2d(positions, self.head_dim)
            q = _apply_token_rotary(q, phases[:, :query_count])
            k = _apply_token_rotary(k, phases)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        allowed = query_valid[:, :, None] & key_valid[:, None, :]
        # Invalid gathered query slots are discarded during scatter, but SDPA
        # must still receive one finite key to avoid an all-masked row.
        invalid_queries = ~query_valid
        if bool(invalid_queries.any()):
            allowed = allowed.clone()
            allowed[:, :, fallback_key_index] |= invalid_queries
        attention_bias = torch.zeros(
            groups,
            1,
            query_count,
            token_count,
            device=tokens.device,
            dtype=q.dtype,
        )
        attention_bias.masked_fill_(~allowed[:, None], -torch.inf)
        if pair_geometry == "offset_valid" and confidence_bias:
            key_bias = (
                confidence.float().clamp_min(1e-4).log()
                * float(confidence_bias)
            ).to(dtype=q.dtype)
            key_bias = torch.where(
                key_valid,
                key_bias,
                torch.zeros_like(key_bias),
            )
            attention_bias = attention_bias + key_bias[:, None, None, :]

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(
            groups, query_count, channels
        )
        output = self.proj_drop(self.proj_local(output))
        return output * query_valid[:, :, None].to(output.dtype)


class LocalReferenceFusionBlock(nn.Module):
    """One local fusion transformer block; its MLP is intentionally mandatory."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.norm1 = RMSNorm(self.hidden_size, eps=1e-6)
        self.fusion_attention = LocalReferenceFusionAttention(
            self.hidden_size, num_heads, dropout
        )
        self.norm2 = RMSNorm(self.hidden_size, eps=1e-6)
        self.fusion_mlp = FeedForward(
            self.hidden_size, int(self.hidden_size * float(mlp_ratio))
        )
        self.fusion_modulation = nn.Sequential(
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        query_count: int,
        positions: torch.Tensor,
        key_valid: torch.Tensor,
        query_valid: torch.Tensor,
        confidence: torch.Tensor,
        pair_geometry: str,
        confidence_bias: float,
        fallback_key_index: int,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.fusion_modulation(condition).chunk(6, dim=-1)
        normalized = apply_adaln(self.norm1(tokens), shift_msa, scale_msa)
        attention = self.fusion_attention(
            normalized,
            query_count,
            positions,
            key_valid,
            query_valid,
            confidence,
            pair_geometry,
            confidence_bias,
            fallback_key_index,
        )
        output = tokens[:, :query_count] + gate_msa[:, :query_count] * attention
        output = output + gate_mlp[:, :query_count] * self.fusion_mlp(
            apply_adaln(
                self.norm2(output),
                shift_mlp[:, :query_count],
                scale_mlp[:, :query_count],
            )
        )
        return output


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
        reference_fusion_enabled: bool = False,
        reference_fusion_num_blocks: int = 1,
        reference_fusion_window_size: int = 3,
        reference_fusion_attention_mode: str = "target_only",
        reference_fusion_pair_geometry: str = "offset_valid",
        reference_fusion_mlp_ratio: float = 4.0,
        reference_fusion_dropout: float = 0.0,
        reference_fusion_chunk_size: int = 64,
        reference_fusion_confidence_bias: float = 1.0,
        reference_fusion_ref_timestep: str = "shared",
        reference_fusion_init: str = "copy",
        reference_fusion_init_block: int = 0,
        reference_fusion_zero_init_gates: bool = True,
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
        self.reference_fusion_enabled = bool(reference_fusion_enabled)
        self.reference_fusion_num_blocks = int(reference_fusion_num_blocks)
        self.reference_fusion_window_size = int(reference_fusion_window_size)
        self.reference_fusion_attention_mode = str(
            reference_fusion_attention_mode
        )
        self.reference_fusion_pair_geometry = str(
            reference_fusion_pair_geometry
        )
        self.reference_fusion_chunk_size = int(reference_fusion_chunk_size)
        self.reference_fusion_confidence_bias = float(
            reference_fusion_confidence_bias
        )
        self.reference_fusion_ref_timestep = str(
            reference_fusion_ref_timestep
        )
        self.reference_fusion_init = str(reference_fusion_init)
        self.reference_fusion_init_block = int(reference_fusion_init_block)
        self.reference_fusion_zero_init_gates = bool(
            reference_fusion_zero_init_gates
        )
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0 to retain the pixel pathway")
        if self.reference_fusion_enabled:
            if self.reference_fusion_num_blocks < 1:
                raise ValueError("reference_fusion_num_blocks must be >= 1")
            if (
                self.reference_fusion_window_size < 1
                or self.reference_fusion_window_size % 2 == 0
            ):
                raise ValueError(
                    "reference_fusion_window_size must be a positive odd integer"
                )
            if self.reference_fusion_attention_mode not in (
                "target_only",
                "joint",
            ):
                raise ValueError(
                    "reference_fusion_attention_mode must be target_only|joint"
                )
            if self.reference_fusion_pair_geometry not in (
                "none",
                "offset_valid",
            ):
                raise ValueError(
                    "reference_fusion_pair_geometry must be none|offset_valid"
                )
            if self.reference_fusion_chunk_size < 1:
                raise ValueError("reference_fusion_chunk_size must be >= 1")
            if self.reference_fusion_ref_timestep not in ("shared", "zero"):
                raise ValueError(
                    "reference_fusion_ref_timestep must be shared|zero"
                )
            if self.reference_fusion_init not in ("copy", "random"):
                raise ValueError("reference_fusion_init must be copy|random")
            if not 0 <= self.reference_fusion_init_block < self.patch_depth:
                raise ValueError(
                    "reference_fusion_init_block must index an existing patch block"
                )

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

        self.reference_fusion_blocks = nn.ModuleList()
        if self.reference_fusion_enabled:
            self.reference_fusion_blocks = nn.ModuleList(
                [
                    LocalReferenceFusionBlock(
                        self.hidden_size,
                        self.num_groups,
                        reference_fusion_mlp_ratio,
                        reference_fusion_dropout,
                    )
                    for _ in range(self.reference_fusion_num_blocks)
                ]
            )

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
        self.precompute_reference_windows = dict()
        self.last_repa_tokens = None

        self.initialize_weights()
        if (
            self.reference_fusion_enabled
            and self.reference_fusion_zero_init_gates
        ):
            self._zero_reference_fusion_gates()
        if self.reference_fusion_enabled:
            self.register_load_state_dict_pre_hook(
                self._reference_fusion_load_pre_hook
            )

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
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _zero_reference_fusion_gates(self):
        """Zero only the two residual-gate chunks of each fusion AdaLN."""
        hidden = self.hidden_size
        for block in self.reference_fusion_blocks:
            linear = block.fusion_modulation[0]
            with torch.no_grad():
                linear.weight[2 * hidden:3 * hidden].zero_()
                linear.bias[2 * hidden:3 * hidden].zero_()
                linear.weight[5 * hidden:6 * hidden].zero_()
                linear.bias[5 * hidden:6 * hidden].zero_()

    def _reference_fusion_load_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Seed missing fusion weights from one pretrained image patch block.

        A checkpoint produced by this architecture already contains fusion
        keys and is left untouched. A legacy PixelDiT checkpoint does not, so
        its image-side block weights are translated into every dedicated local
        block immediately before loading. This works for training, validation,
        and standalone inference without a train-loop-only initialization hook.
        """
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        if (
            not self.reference_fusion_enabled
            or self.reference_fusion_init != "copy"
        ):
            return
        probe = prefix + "reference_fusion_blocks.0.norm1.weight"
        if probe in state_dict:
            return

        source = prefix + (
            f"patch_blocks.{self.reference_fusion_init_block}."
        )
        translations = {
            "norm1.weight": "norm_x1.weight",
            "fusion_attention.qkv_local.weight": "attn.qkv_x.weight",
            "fusion_attention.q_norm.weight": "attn.q_norm_x.weight",
            "fusion_attention.k_norm.weight": "attn.k_norm_x.weight",
            "fusion_attention.proj_local.weight": "attn.proj_x.weight",
            "fusion_attention.proj_local.bias": "attn.proj_x.bias",
            "norm2.weight": "norm_x2.weight",
            "fusion_mlp.w1.weight": "mlp_x.w1.weight",
            "fusion_mlp.w2.weight": "mlp_x.w2.weight",
            "fusion_mlp.w3.weight": "mlp_x.w3.weight",
            "fusion_modulation.0.weight": "adaLN_modulation_img.0.weight",
            "fusion_modulation.0.bias": "adaLN_modulation_img.0.bias",
        }
        source_keys = [source + suffix for suffix in translations.values()]
        if any(key not in state_dict for key in source_keys):
            # Leave the constructor initialization in place. load_state_dict
            # will report the fusion keys as missing, which is safer than
            # silently performing a partial copy.
            return

        hidden = self.hidden_size
        for block_index in range(self.reference_fusion_num_blocks):
            destination = prefix + f"reference_fusion_blocks.{block_index}."
            for destination_suffix, source_suffix in translations.items():
                value = state_dict[source + source_suffix].clone()
                if (
                    self.reference_fusion_zero_init_gates
                    and destination_suffix
                    in (
                        "fusion_modulation.0.weight",
                        "fusion_modulation.0.bias",
                    )
                ):
                    value[2 * hidden:3 * hidden].zero_()
                    value[5 * hidden:6 * hidden].zero_()
                state_dict[destination + destination_suffix] = value

    def _fetch_reference_window_layout(self, height, width, device):
        """Patch indices/validity for every target-centred local window."""
        key = (height, width, self.reference_fusion_window_size)
        cached = self.precompute_reference_windows.get(key)
        if cached is None:
            radius = self.reference_fusion_window_size // 2
            offsets_y, offsets_x = torch.meshgrid(
                torch.arange(-radius, radius + 1, dtype=torch.long),
                torch.arange(-radius, radius + 1, dtype=torch.long),
                indexing="ij",
            )
            offsets = torch.stack(
                [offsets_x.reshape(-1), offsets_y.reshape(-1)], dim=-1
            )
            centres_y, centres_x = torch.meshgrid(
                torch.arange(height, dtype=torch.long),
                torch.arange(width, dtype=torch.long),
                indexing="ij",
            )
            centres = torch.stack(
                [centres_x.reshape(-1), centres_y.reshape(-1)], dim=-1
            )
            candidates = centres[:, None, :] + offsets[None, :, :]
            valid = (
                (candidates[..., 0] >= 0)
                & (candidates[..., 0] < width)
                & (candidates[..., 1] >= 0)
                & (candidates[..., 1] < height)
            )
            clamped_x = candidates[..., 0].clamp(0, width - 1)
            clamped_y = candidates[..., 1].clamp(0, height - 1)
            indices = clamped_y * width + clamped_x
            cached = (indices, valid, offsets)
            self.precompute_reference_windows[key] = cached
        return tuple(item.to(device) for item in cached)

    @staticmethod
    def _prepare_reference_fusion_refs(
        refs: torch.Tensor,
        render_batch: int,
        height: int,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if refs.dim() != 5:
            raise ValueError(
                "reference_fusion_refs must be [B,K,C,H,W], "
                f"got {tuple(refs.shape)}"
            )
        if refs.shape[1] < 1:
            raise ValueError("reference_fusion_refs needs at least one reference")
        if refs.shape[-2:] != (height, width):
            raise ValueError(
                f"reference size {tuple(refs.shape[-2:])} does not match "
                f"render size {(height, width)}"
            )
        if render_batch % refs.shape[0]:
            raise ValueError(
                f"reference batch {refs.shape[0]} does not divide "
                f"render batch {render_batch}"
            )
        if refs.shape[0] != render_batch:
            refs = refs.repeat(
                render_batch // refs.shape[0], 1, 1, 1, 1
            )
        return refs.to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_reference_fusion_geometry(
        geometry: torch.Tensor,
        render_batch: int,
        num_refs: int,
        token_count: int,
        device,
    ) -> torch.Tensor:
        if geometry.dim() != 4 or geometry.shape[-1] != 4:
            raise ValueError(
                "reference_fusion_geometry must be [B,K,L,4], "
                f"got {tuple(geometry.shape)}"
            )
        if geometry.shape[1] != num_refs:
            raise ValueError(
                f"fusion geometry has {geometry.shape[1]} references, "
                f"expected {num_refs}"
            )
        if geometry.shape[2] != token_count:
            raise ValueError(
                f"fusion geometry token count {geometry.shape[2]} != "
                f"{token_count}"
            )
        if render_batch % geometry.shape[0]:
            raise ValueError(
                f"fusion geometry batch {geometry.shape[0]} does not divide "
                f"render batch {render_batch}"
            )
        if geometry.shape[0] != render_batch:
            geometry = geometry.repeat(
                render_batch // geometry.shape[0], 1, 1, 1
            )
        # Keep continuous patch centres in float32. Rounding bf16 centres can
        # move a correspondence by a substantial fraction of one patch.
        return geometry.to(device=device, dtype=torch.float32)

    @staticmethod
    def _prepare_fusion_patch_geometry(
        geometry: torch.Tensor,
        render_batch: int,
        num_refs: int,
        token_count: int,
        device,
        dtype,
    ) -> torch.Tensor:
        views = 1 + num_refs
        if geometry.dim() == 3:
            if geometry.shape[0] % views:
                raise ValueError(
                    f"patch geometry rows {geometry.shape[0]} are not "
                    f"divisible by {views} views"
                )
            geometry = geometry.reshape(
                geometry.shape[0] // views,
                views,
                geometry.shape[1],
                geometry.shape[2],
            )
        if geometry.dim() != 4 or geometry.shape[1] != views:
            raise ValueError(
                "fusion patch geometry must be [B,1+K,L,F], "
                f"got {tuple(geometry.shape)}"
            )
        if geometry.shape[2] != token_count:
            raise ValueError(
                f"patch geometry token count {geometry.shape[2]} != "
                f"{token_count}"
            )
        if render_batch % geometry.shape[0]:
            raise ValueError(
                f"patch geometry batch {geometry.shape[0]} does not divide "
                f"render batch {render_batch}"
            )
        if geometry.shape[0] != render_batch:
            geometry = geometry.repeat(
                render_batch // geometry.shape[0], 1, 1, 1
            )
        return geometry.to(device=device, dtype=dtype)

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

    def _run_reference_fusion(
        self,
        render_tokens: torch.Tensor,
        reference_tokens: torch.Tensor,
        render_condition: torch.Tensor,
        reference_condition: torch.Tensor,
        fusion_geometry: torch.Tensor,
        patch_height: int,
        patch_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run correspondence-centred local blocks and merge overlaps by mean.

        Every target patch defines one group containing its target kxk window
        plus one correspondence-centred kxk window per reference. ``joint``
        performs full attention over every token in the group and scatters
        updates back to both render and reference grids. ``target_only`` uses
        the same keys/values but queries and updates only the target window.

        We aggregate residual deltas rather than complete token values. Thus
        zero-initialized gates are an exact no-op even when one patch appears
        in several overlapping groups.
        """
        batch, token_count, channels = render_tokens.shape
        ref_batch, num_refs, ref_length, ref_channels = reference_tokens.shape
        if (ref_batch, ref_length, ref_channels) != (
            batch,
            token_count,
            channels,
        ):
            raise ValueError(
                "reference token shape must be [B,K,L,D] matching render "
                f"tokens; got {tuple(reference_tokens.shape)} versus "
                f"{tuple(render_tokens.shape)}"
            )
        if fusion_geometry.shape != (batch, num_refs, token_count, 4):
            raise ValueError(
                "fusion geometry shape must be [B,K,L,4], got "
                f"{tuple(fusion_geometry.shape)}"
            )
        if token_count != patch_height * patch_width:
            raise ValueError(
                f"token count {token_count} != patch grid "
                f"{patch_height}x{patch_width}"
            )

        window_indices, window_valid, offsets = (
            self._fetch_reference_window_layout(
                patch_height, patch_width, render_tokens.device
            )
        )
        window_tokens = offsets.shape[0]
        fallback_key_index = window_tokens // 2
        x_scale = 16.0 / max(patch_width - 1, 1)
        y_scale = 16.0 / max(patch_height - 1, 1)
        target_relative_positions = torch.stack(
            [
                offsets[:, 0].float() * x_scale,
                offsets[:, 1].float() * y_scale,
            ],
            dim=-1,
        )
        joint = self.reference_fusion_attention_mode == "joint"
        chunk_size = self.reference_fusion_chunk_size
        batch_offsets = (
            torch.arange(batch, device=render_tokens.device) * token_count
        )
        reference_view_offsets = (
            torch.arange(num_refs, device=render_tokens.device) * token_count
        )
        reference_batch_offsets = (
            torch.arange(batch, device=render_tokens.device)
            * num_refs
            * token_count
        )

        for block in self.reference_fusion_blocks:
            render_delta_sum = torch.zeros_like(render_tokens).reshape(
                batch * token_count, channels
            )
            render_count = torch.zeros(
                batch * token_count,
                1,
                device=render_tokens.device,
                dtype=torch.float32,
            )
            if joint:
                reference_delta_sum = torch.zeros_like(
                    reference_tokens
                ).reshape(batch * num_refs * token_count, channels)
                reference_count = torch.zeros(
                    batch * num_refs * token_count,
                    1,
                    device=render_tokens.device,
                    dtype=torch.float32,
                )

            reference_flat = reference_tokens.reshape(
                batch, num_refs * token_count, channels
            )
            reference_condition_flat = reference_condition.reshape(
                batch, num_refs * token_count, channels
            )

            for start in range(0, token_count, chunk_size):
                stop = min(start + chunk_size, token_count)
                centre_ids = torch.arange(
                    start, stop, device=render_tokens.device
                )
                centres_in_chunk = stop - start
                target_indices = window_indices[centre_ids]
                target_valid = window_valid[centre_ids]

                target_group = render_tokens[:, target_indices]
                target_condition_group = render_condition[:, target_indices]
                target_group_valid = target_valid[None].expand(
                    batch, -1, -1
                )

                centre_geometry = fusion_geometry[
                    :, :, centre_ids, :
                ].permute(0, 2, 1, 3)
                centre_x = centre_geometry[..., 0]
                centre_y = centre_geometry[..., 1]
                centre_confidence = centre_geometry[..., 2]
                centre_valid = (
                    (centre_geometry[..., 3] > 0.5)
                    & torch.isfinite(centre_x)
                    & torch.isfinite(centre_y)
                    & torch.isfinite(centre_confidence)
                )
                safe_centre_x = torch.where(
                    torch.isfinite(centre_x),
                    centre_x,
                    torch.zeros_like(centre_x),
                )
                safe_centre_y = torch.where(
                    torch.isfinite(centre_y),
                    centre_y,
                    torch.zeros_like(centre_y),
                )
                centre_confidence = torch.where(
                    torch.isfinite(centre_confidence),
                    centre_confidence.clamp(0, 1),
                    torch.zeros_like(centre_confidence),
                )
                base_x = safe_centre_x.round().long()
                base_y = safe_centre_y.round().long()
                candidate_x = base_x[..., None] + offsets[:, 0]
                candidate_y = base_y[..., None] + offsets[:, 1]
                reference_valid = (
                    centre_valid[..., None]
                    & (candidate_x >= 0)
                    & (candidate_x < patch_width)
                    & (candidate_y >= 0)
                    & (candidate_y < patch_height)
                )
                clamped_x = candidate_x.clamp(0, patch_width - 1)
                clamped_y = candidate_y.clamp(0, patch_height - 1)
                reference_indices = (
                    clamped_y * patch_width + clamped_x
                )
                reference_global_indices = (
                    reference_indices
                    + reference_view_offsets[None, None, :, None]
                )
                reference_gather_indices = reference_global_indices.reshape(
                    batch, -1
                )
                reference_group = torch.gather(
                    reference_flat,
                    1,
                    reference_gather_indices[:, :, None].expand(
                        -1, -1, channels
                    ),
                ).reshape(
                    batch,
                    centres_in_chunk,
                    num_refs,
                    window_tokens,
                    channels,
                )
                reference_condition_group = torch.gather(
                    reference_condition_flat,
                    1,
                    reference_gather_indices[:, :, None].expand(
                        -1, -1, channels
                    ),
                ).reshape(
                    batch,
                    centres_in_chunk,
                    num_refs,
                    window_tokens,
                    channels,
                )

                group_tokens = torch.cat(
                    [
                        target_group,
                        reference_group.reshape(
                            batch,
                            centres_in_chunk,
                            num_refs * window_tokens,
                            channels,
                        ),
                    ],
                    dim=2,
                )
                group_condition = torch.cat(
                    [
                        target_condition_group,
                        reference_condition_group.reshape(
                            batch,
                            centres_in_chunk,
                            num_refs * window_tokens,
                            channels,
                        ),
                    ],
                    dim=2,
                )
                group_valid = torch.cat(
                    [
                        target_group_valid,
                        reference_valid.reshape(
                            batch,
                            centres_in_chunk,
                            num_refs * window_tokens,
                        ),
                    ],
                    dim=2,
                )
                target_positions = target_relative_positions[
                    None, None
                ].expand(batch, centres_in_chunk, -1, -1)
                reference_positions = torch.stack(
                    [
                        (candidate_x.float() - safe_centre_x[..., None])
                        * x_scale,
                        (candidate_y.float() - safe_centre_y[..., None])
                        * y_scale,
                    ],
                    dim=-1,
                )
                group_positions = torch.cat(
                    [
                        target_positions,
                        reference_positions.reshape(
                            batch,
                            centres_in_chunk,
                            num_refs * window_tokens,
                            2,
                        ),
                    ],
                    dim=2,
                )
                group_confidence = torch.cat(
                    [
                        torch.ones(
                            batch,
                            centres_in_chunk,
                            window_tokens,
                            device=render_tokens.device,
                            dtype=fusion_geometry.dtype,
                        ),
                        centre_confidence[..., None]
                        .expand(-1, -1, -1, window_tokens)
                        .reshape(
                            batch,
                            centres_in_chunk,
                            num_refs * window_tokens,
                        ),
                    ],
                    dim=2,
                )

                group_token_count = group_tokens.shape[2]
                query_count = (
                    group_token_count if joint else window_tokens
                )
                flattened_groups = batch * centres_in_chunk
                updated = block(
                    group_tokens.reshape(
                        flattened_groups, group_token_count, channels
                    ),
                    group_condition.reshape(
                        flattened_groups, group_token_count, channels
                    ),
                    query_count,
                    group_positions.reshape(
                        flattened_groups, group_token_count, 2
                    ),
                    group_valid.reshape(
                        flattened_groups, group_token_count
                    ),
                    group_valid[:, :, :query_count].reshape(
                        flattened_groups, query_count
                    ),
                    group_confidence.reshape(
                        flattened_groups, group_token_count
                    ),
                    self.reference_fusion_pair_geometry,
                    self.reference_fusion_confidence_bias,
                    fallback_key_index,
                ).reshape(
                    batch, centres_in_chunk, query_count, channels
                )

                target_delta = (
                    updated[:, :, :window_tokens] - target_group
                )
                target_scatter_indices = (
                    target_indices[None]
                    + batch_offsets[:, None, None]
                )
                target_scatter_valid = target_group_valid.reshape(-1)
                flat_target_indices = target_scatter_indices.reshape(-1)[
                    target_scatter_valid
                ]
                flat_target_delta = target_delta.reshape(
                    -1, channels
                )[target_scatter_valid]
                render_delta_sum.index_add_(
                    0, flat_target_indices, flat_target_delta
                )
                render_count.index_add_(
                    0,
                    flat_target_indices,
                    torch.ones(
                        flat_target_indices.shape[0],
                        1,
                        device=render_tokens.device,
                        dtype=render_count.dtype,
                    ),
                )

                if joint:
                    reference_updated = updated[
                        :, :, window_tokens:
                    ].reshape(
                        batch,
                        centres_in_chunk,
                        num_refs,
                        window_tokens,
                        channels,
                    )
                    reference_delta = (
                        reference_updated - reference_group
                    )
                    reference_scatter_indices = (
                        reference_indices
                        + reference_view_offsets[None, None, :, None]
                        + reference_batch_offsets[:, None, None, None]
                    )
                    reference_scatter_valid = reference_valid.reshape(-1)
                    flat_reference_indices = (
                        reference_scatter_indices.reshape(-1)[
                            reference_scatter_valid
                        ]
                    )
                    flat_reference_delta = reference_delta.reshape(
                        -1, channels
                    )[reference_scatter_valid]
                    reference_delta_sum.index_add_(
                        0, flat_reference_indices, flat_reference_delta
                    )
                    reference_count.index_add_(
                        0,
                        flat_reference_indices,
                        torch.ones(
                            flat_reference_indices.shape[0],
                            1,
                            device=render_tokens.device,
                            dtype=reference_count.dtype,
                        ),
                    )

            render_count = render_count.reshape(batch, token_count, 1)
            averaged_render_delta = render_delta_sum.reshape(
                batch, token_count, channels
            ) / render_count.clamp_min(1).to(render_tokens.dtype)
            render_tokens = render_tokens + torch.where(
                render_count > 0,
                averaged_render_delta,
                torch.zeros_like(averaged_render_delta),
            )
            if joint:
                reference_count = reference_count.reshape(
                    batch, num_refs, token_count, 1
                )
                averaged_reference_delta = reference_delta_sum.reshape(
                    batch, num_refs, token_count, channels
                ) / reference_count.clamp_min(1).to(reference_tokens.dtype)
                reference_tokens = reference_tokens + torch.where(
                    reference_count > 0,
                    averaged_reference_delta,
                    torch.zeros_like(averaged_reference_delta),
                )

        return render_tokens, reference_tokens

    def forward(
        self,
        x,
        t,
        y=None,
        s=None,
        mask=None,
        num_views=1,
        patch_geometry=None,
        reference_fusion_refs=None,
        reference_fusion_geometry=None,
    ):
        """``num_views`` > 1 means ``x`` holds V views per sample, stacked **b-major** on the
        batch axis (row ``b*V + v``); see ``merge_views``. The views meet only inside the patch
        blocks' joint attention. Everything else -- the pixel blocks, ``final_layer``, the
        returned shape -- treats them as independent rows, so the output stays ``[B*V,3,H,W]``
        and the caller reshapes if it wants a view axis. ``num_views=1`` is a bit-exact no-op.

        ``t`` is per row (length ``B*V``); ``y`` and ``mask`` are per SAMPLE (length ``B``),
        because the text stream is shared across a sample's views.

        With local reference fusion active, ``x`` contains render rows only
        (``num_views=1``). Clean references arrive separately as
        ``reference_fusion_refs`` and are discarded after the dedicated local
        blocks, so the existing global patch and pixel pathways remain render-only.
        """
        B, _, H, W = x.shape          # B here is B*V: one row per (sample, view)
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws
        fusion_active = (
            self.reference_fusion_enabled
            and reference_fusion_refs is not None
        )
        if reference_fusion_refs is not None and not self.reference_fusion_enabled:
            raise ValueError(
                "reference_fusion_refs were supplied but local reference fusion "
                "is disabled in the model"
            )
        if reference_fusion_geometry is not None and not fusion_active:
            raise ValueError(
                "reference_fusion_geometry requires reference_fusion_refs"
            )
        if fusion_active and num_views != 1:
            raise ValueError(
                "local reference fusion keeps references outside diffusion rows; "
                f"num_views must be 1, got {num_views}"
            )

        prepared_refs = prepared_fusion_geometry = None
        num_refs = 0
        if fusion_active:
            if reference_fusion_geometry is None:
                raise ValueError(
                    "local reference fusion requires correspondence geometry"
                )
            prepared_refs = self._prepare_reference_fusion_refs(
                reference_fusion_refs,
                B,
                H,
                W,
                x.device,
                x.dtype,
            )
            num_refs = prepared_refs.shape[1]
            prepared_fusion_geometry = (
                self._prepare_reference_fusion_geometry(
                    reference_fusion_geometry,
                    B,
                    num_refs,
                    L,
                    x.device,
                )
            )

        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        text_on = y is not None
        patch_num_views = 1 if fusion_active else num_views
        y_emb, Ltxt, Btxt = None, 0, B // patch_num_views
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
            reference_s0 = None
            if fusion_active:
                ref_rows = prepared_refs.reshape(
                    B * num_refs, prepared_refs.shape[2], H, W
                )
                ref_patches = torch.nn.functional.unfold(
                    ref_rows,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ).transpose(1, 2)
                reference_s0 = self.s_embedder(ref_patches).reshape(
                    B, num_refs, L, self.hidden_size
                )
                if (
                    self.patch_geometry_embedder is not None
                    and patch_geometry is not None
                ):
                    geometry = self._prepare_fusion_patch_geometry(
                        patch_geometry,
                        B,
                        num_refs,
                        L,
                        x.device,
                        s0.dtype,
                    )
                    geometry_rows = geometry.reshape(
                        B * (1 + num_refs), L, geometry.shape[-1]
                    )
                    geometry_embedding = self.patch_geometry_embedder(
                        geometry_rows, 1 + num_refs
                    ).reshape(B, 1 + num_refs, L, self.hidden_size)
                    s0 = s0 + geometry_embedding[:, 0]
                    reference_s0 = (
                        reference_s0 + geometry_embedding[:, 1:]
                    )

                reference_timestep = t.view(B, 1).expand(B, num_refs)
                if self.reference_fusion_ref_timestep == "zero":
                    reference_timestep = torch.zeros_like(
                        reference_timestep
                    )
                reference_t_emb = self.t_embedder(
                    reference_timestep.reshape(-1)
                ).reshape(B, num_refs, 1, self.hidden_size)
                render_condition = condition.expand(-1, L, -1)
                reference_condition = torch.nn.functional.silu(
                    reference_t_emb
                ).expand(-1, -1, L, -1)
                if self.training and getattr(
                    self, "grad_checkpointing", False
                ):
                    def _checkpointed_reference_fusion(
                        render_tokens,
                        reference_tokens,
                        render_cond,
                        reference_cond,
                        fusion_geometry,
                    ):
                        return self._run_reference_fusion(
                            render_tokens,
                            reference_tokens,
                            render_cond,
                            reference_cond,
                            fusion_geometry,
                            Hs,
                            Ws,
                        )[0]

                    s0 = checkpoint(
                        _checkpointed_reference_fusion,
                        s0,
                        reference_s0,
                        render_condition,
                        reference_condition,
                        prepared_fusion_geometry,
                        use_reentrant=False,
                    )
                    reference_s0 = None
                else:
                    s0, reference_s0 = self._run_reference_fusion(
                        s0,
                        reference_s0,
                        render_condition,
                        reference_condition,
                        prepared_fusion_geometry,
                        Hs,
                        Ws,
                    )
            elif (
                self.patch_geometry_embedder is not None
                and patch_geometry is not None
            ):
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
                    pad_img = torch.zeros((Btxt, patch_num_views * L), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(
                        Btxt, 1, 1, Ltxt + patch_num_views * L
                    )
            self.last_repa_tokens = None
            s = s0
            for i in range(self.patch_depth):
                s, y_emb = self.patch_blocks[i](
                    s,
                    y_emb,
                    condition,
                    pos,
                    pos_txt,
                    attn_mask_joint,
                    patch_num_views,
                )
                if 0 < self.repa_encoder_index == (i + 1):
                    self.last_repa_tokens = s
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
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
