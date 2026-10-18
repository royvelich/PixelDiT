import torch
import torch.nn as nn
import torch.nn.functional as F

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
from .prope import PropeDotProductAttention, invert_se3


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


def build_plucker_ray_patches(
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return PixelUnshuffle-equivalent Plucker features ``[B*V,L,6*P*P]``.

    ``viewmats`` are camera-from-target transforms and ``intrinsics`` use
    PRoPE's normalized image coordinates. Rays are expressed in the render
    camera's coordinate frame, so every view shares one canonical frame.
    """
    if viewmats.dim() != 4 or viewmats.shape[-2:] != (4, 4):
        raise ValueError("camera view matrices must be [B,V,4,4]")
    if intrinsics.shape != viewmats.shape[:2] + (3, 3):
        raise ValueError("camera intrinsics must be [B,V,3,3]")
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image {(height, width)} is not divisible by patch size {patch_size}"
        )
    device = viewmats.device
    with torch.autocast(device_type=device.type, enabled=False):
        w2cs = viewmats.float()
        Ks = intrinsics.float()
        c2ws = invert_se3(w2cs)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pixels = torch.stack(
            [
                xx / float(width) - 0.5,
                yy / float(height) - 0.5,
                torch.ones_like(xx),
            ],
            dim=-1,
        )
        camera_rays = torch.einsum(
            "bvij,hwj->bvhwi", torch.linalg.inv(Ks), pixels
        )
        directions = F.normalize(
            torch.einsum(
                "bvij,bvhwj->bvhwi", c2ws[..., :3, :3], camera_rays
            ),
            dim=-1,
        )
        origins = c2ws[..., :3, 3].unsqueeze(-2).unsqueeze(-2)
        origins = origins.expand_as(directions)
        moments = torch.cross(origins, directions, dim=-1)
        raymaps = torch.cat([moments, directions], dim=-1)
        batch, views = raymaps.shape[:2]
        raymaps = raymaps.permute(0, 1, 4, 2, 3).reshape(
            batch * views, 6, height, width
        )
        patches = F.unfold(
            raymaps, kernel_size=patch_size, stride=patch_size
        ).transpose(1, 2)
    return patches.to(dtype=dtype)


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
        self.prope_ref_k = None
        self.prope_ref_v = None
        self.prope_ref_k_norm = None
        self.prope_source = None
        self.prope_references = None

    def enable_prope_reference_attention(self, freq_base: float = 100.0) -> None:
        """Add ArtiFixer-style render-query/reference-KV attention."""
        if self.prope_ref_k is not None:
            return
        self.prope_ref_k = nn.Linear(self.dim, self.dim, bias=True)
        self.prope_ref_v = nn.Linear(self.dim, self.dim, bias=True)
        self.prope_ref_k_norm = RMSNorm(self.head_dim)
        self.prope_source = PropeDotProductAttention(
            self.head_dim, freq_base=freq_base
        )
        self.prope_references = PropeDotProductAttention(
            self.head_dim, freq_base=freq_base
        )
        nn.init.zeros_(self.prope_ref_v.weight)
        nn.init.zeros_(self.prope_ref_v.bias)

    def _prope_reference_attention(
        self,
        x: torch.Tensor,
        render_query: torch.Tensor,
        num_views: int,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        patch_grid,
        camera_keep: torch.Tensor,
    ) -> torch.Tensor:
        """Render queries attend to every reference token; refs get zero output."""
        if num_views <= 1:
            return torch.zeros_like(x)
        batch_rows, tokens, channels = x.shape
        if batch_rows % num_views:
            raise ValueError(
                f"image rows {batch_rows} are not divisible by num_views={num_views}"
            )
        batch = batch_rows // num_views
        if viewmats.shape != (batch, num_views, 4, 4):
            raise ValueError(
                "PRoPE view matrices must be "
                f"{(batch, num_views, 4, 4)}, got {tuple(viewmats.shape)}"
            )
        if intrinsics.shape != (batch, num_views, 3, 3):
            raise ValueError(
                "PRoPE intrinsics must be "
                f"{(batch, num_views, 3, 3)}, got {tuple(intrinsics.shape)}"
            )
        if patch_grid is None or len(patch_grid) != 2:
            raise ValueError("PRoPE reference attention requires a patch grid")
        patch_height, patch_width = (int(value) for value in patch_grid)
        if patch_height * patch_width != tokens:
            raise ValueError(
                f"patch grid {(patch_height, patch_width)} has "
                f"{patch_height * patch_width} tokens, expected {tokens}"
            )

        references = x.view(batch, num_views, tokens, channels)[:, 1:]
        num_refs = num_views - 1
        key = self.prope_ref_k(references).view(
            batch, num_refs, tokens, self.num_heads, self.head_dim
        )
        value = self.prope_ref_v(references).view(
            batch, num_refs, tokens, self.num_heads, self.head_dim
        )
        key = self.prope_ref_k_norm(key)
        key = key.permute(0, 3, 1, 2, 4).reshape(
            batch, self.num_heads, num_refs * tokens, self.head_dim
        )
        value = value.permute(0, 3, 1, 2, 4).reshape(
            batch, self.num_heads, num_refs * tokens, self.head_dim
        )
        query = render_query.view(
            batch, num_views, self.num_heads, tokens, self.head_dim
        )[:, 0]

        self.prope_source.update_coeffs(
            patch_width, patch_height, query.device
        )
        self.prope_references.update_coeffs(
            patch_width, patch_height, query.device
        )
        self.prope_source.prepare(viewmats[:, :1], intrinsics[:, :1])
        self.prope_references.prepare(viewmats[:, 1:], intrinsics[:, 1:])

        attention_dtype = query.dtype
        query = self.prope_source.apply_to_q(query).to(attention_dtype)
        key = self.prope_references.apply_to_kv(key).to(attention_dtype)
        value = self.prope_references.apply_to_kv(value).to(attention_dtype)
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0
        )
        output = self.prope_source.apply_to_output(output).to(attention_dtype)
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        if camera_keep is not None:
            if camera_keep.shape != (batch, 1, 1):
                raise ValueError(
                    "camera_keep must be "
                    f"{(batch, 1, 1)}, got {tuple(camera_keep.shape)}"
                )
            output = output * camera_keep.to(output.dtype)
        zeros = output.new_zeros(batch, num_refs, tokens, channels)
        return torch.cat([output[:, None], zeros], dim=1).reshape_as(x)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            pos_img: torch.Tensor,
            pos_txt: torch.Tensor = None,
            attn_mask: torch.Tensor = None,
            num_views: int = 1,
            prope_w2cs: torch.Tensor = None,
            prope_intrinsics: torch.Tensor = None,
            patch_grid=None,
            camera_keep: torch.Tensor = None,
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
        render_query = qx

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
        if self.prope_ref_k is not None:
            if prope_w2cs is None or prope_intrinsics is None:
                raise ValueError(
                    "PRoPE reference attention requires camera matrices"
                )
            out_x = out_x + self._prope_reference_attention(
                x,
                render_query,
                num_views,
                prope_w2cs,
                prope_intrinsics,
                patch_grid,
                camera_keep,
            )
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
        self.camera_ray_embedder = None

    def enable_camera_conditioning(
        self,
        ray_patch_dim: int,
        rays_enabled: bool,
        prope_enabled: bool,
        prope_freq_base: float,
    ) -> None:
        if rays_enabled:
            self.camera_ray_embedder = nn.Linear(
                int(ray_patch_dim), self.hidden_size, bias=True
            )
            nn.init.zeros_(self.camera_ray_embedder.weight)
            nn.init.zeros_(self.camera_ray_embedder.bias)
        if prope_enabled:
            self.attn.enable_prope_reference_attention(prope_freq_base)

    def forward(
        self,
        x,
        y,
        c,
        pos_img,
        pos_txt=None,
        attn_mask=None,
        num_views=1,
        camera_ray_patches=None,
        camera_ray_keep=None,
        prope_w2cs=None,
        prope_intrinsics=None,
        patch_grid=None,
        camera_keep=None,
    ):
        # `c` is per IMAGE row, i.e. at B*V. The text stream is at B, so its modulation must be
        # fed the render rows only: b-major rows mean sample b's render is row b*V, which is
        # exactly `c[::num_views]`. A no-op at num_views=1.
        text_on = y is not None
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        if text_on:
            c_txt = c if num_views == 1 else c[::num_views]
            shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c_txt).chunk(6, dim=-1)

        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        if self.camera_ray_embedder is not None:
            if camera_ray_patches is None:
                raise ValueError(
                    "camera-ray conditioning requires camera_ray_patches"
                )
            ray_embedding = self.camera_ray_embedder(camera_ray_patches)
            if camera_ray_keep is not None:
                ray_embedding = ray_embedding * camera_ray_keep.to(
                    ray_embedding.dtype
                )
            x_norm = x_norm + ray_embedding
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y) if text_on else None
        attn_x, attn_y = self.attn(
            x_norm,
            y_norm,
            pos_img,
            pos_txt,
            attn_mask,
            num_views,
            prope_w2cs=prope_w2cs,
            prope_intrinsics=prope_intrinsics,
            patch_grid=patch_grid,
            camera_keep=camera_keep,
        )
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
        camera_conditioning_mode: str = "none",
        camera_ray_rows: str = "all",
        camera_conditioning_dropout: float = 0.0,
        prope_freq_base: float = 100.0,
        pit_view_attention: str = "separate",
        pit_camera_conditioning_mode: str = "none",
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
        self.camera_conditioning_mode = str(camera_conditioning_mode)
        if self.camera_conditioning_mode not in (
            "none",
            "rays",
            "prope",
            "rays_prope",
        ):
            raise ValueError(
                "camera_conditioning_mode must be "
                "none|rays|prope|rays_prope"
            )
        self.camera_rays_enabled = self.camera_conditioning_mode in (
            "rays",
            "rays_prope",
        )
        self.camera_prope_enabled = self.camera_conditioning_mode in (
            "prope",
            "rays_prope",
        )
        self.pit_view_attention = str(pit_view_attention)
        if self.pit_view_attention not in ("separate", "joint"):
            raise ValueError(
                "pit_view_attention must be separate|joint"
            )
        self.pit_camera_conditioning_mode = str(
            pit_camera_conditioning_mode
        )
        if self.pit_camera_conditioning_mode not in (
            "none",
            "rays",
            "prope",
            "rays_prope",
        ):
            raise ValueError(
                "pit_camera_conditioning_mode must be "
                "none|rays|prope|rays_prope"
            )
        self.pit_camera_rays_enabled = (
            self.pit_camera_conditioning_mode in ("rays", "rays_prope")
        )
        self.pit_camera_prope_enabled = (
            self.pit_camera_conditioning_mode in ("prope", "rays_prope")
        )
        self.camera_ray_rows = str(camera_ray_rows)
        if self.camera_ray_rows not in ("all", "render"):
            raise ValueError("camera_ray_rows must be all|render")
        self.camera_conditioning_dropout = float(camera_conditioning_dropout)
        if not (0.0 <= self.camera_conditioning_dropout < 1.0):
            raise ValueError("camera_conditioning_dropout must be in [0,1)")
        self.prope_freq_base = float(prope_freq_base)
        if self.prope_freq_base <= 0:
            raise ValueError("prope_freq_base must be positive")
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
                    view_attention=self.pit_view_attention,
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

        self.initialize_weights()
        if (
            self.camera_conditioning_mode != "none"
            or self.pit_camera_conditioning_mode != "none"
        ):
            # New modules must not perturb initialization of any pre-existing
            # PixelDiT parameter. They are also zero-output at step zero.
            with torch.random.fork_rng(devices=[]):
                if self.camera_conditioning_mode != "none":
                    for block in self.patch_blocks:
                        block.enable_camera_conditioning(
                            ray_patch_dim=(
                                6 * self.patch_size * self.patch_size
                            ),
                            rays_enabled=self.camera_rays_enabled,
                            prope_enabled=self.camera_prope_enabled,
                            prope_freq_base=self.prope_freq_base,
                        )
                if self.pit_camera_conditioning_mode != "none":
                    for block in self.pixel_blocks:
                        block.enable_camera_conditioning(
                            rays_enabled=self.pit_camera_rays_enabled,
                            prope_enabled=self.pit_camera_prope_enabled,
                            prope_freq_base=self.prope_freq_base,
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

    @staticmethod
    def _prepare_camera_conditioning(
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        model_rows: int,
        num_views: int,
        device,
    ):
        if viewmats.dim() == 3:
            viewmats = viewmats.unsqueeze(0)
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(0)
        if viewmats.dim() != 4 or viewmats.shape[1:] != (num_views, 4, 4):
            raise ValueError(
                "camera view matrices must be [B,V,4,4], "
                f"got {tuple(viewmats.shape)}"
            )
        if (
            intrinsics.dim() != 4
            or intrinsics.shape[1:] != (num_views, 3, 3)
        ):
            raise ValueError(
                "camera intrinsics must be [B,V,3,3], "
                f"got {tuple(intrinsics.shape)}"
            )
        if model_rows % num_views:
            raise ValueError(
                f"model rows {model_rows} are not divisible by num_views={num_views}"
            )
        model_batch = model_rows // num_views
        if model_batch % viewmats.shape[0]:
            raise ValueError(
                f"camera batch {viewmats.shape[0]} does not divide "
                f"model batch {model_batch}"
            )
        if intrinsics.shape[0] != viewmats.shape[0]:
            raise ValueError("camera extrinsic and intrinsic batches disagree")
        if viewmats.shape[0] != model_batch:
            repeats = model_batch // viewmats.shape[0]
            viewmats = viewmats.repeat(repeats, 1, 1, 1)
            intrinsics = intrinsics.repeat(repeats, 1, 1, 1)
        return (
            viewmats.to(device=device, dtype=torch.float32),
            intrinsics.to(device=device, dtype=torch.float32),
        )

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
        camera_w2cs=None,
        camera_intrinsics=None,
    ):
        """``num_views`` > 1 means ``x`` holds V views per sample, stacked **b-major** on the
        batch axis (row ``b*V + v``); see ``merge_views``. The views meet only inside the patch
        blocks' joint attention. Pixel blocks keep the historical independent-row path when
        ``pit_view_attention="separate"`` and merge all views onto one attention axis when it is
        ``"joint"``. The final layer and returned shape remain per-row, so the output stays
        ``[B*V,3,H,W]`` and the caller reshapes if it wants a view axis. ``num_views=1`` is a
        bit-exact no-op.

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
        camera_enabled = (
            self.camera_conditioning_mode != "none"
            or self.pit_camera_conditioning_mode != "none"
        )
        if camera_enabled:
            if camera_w2cs is None or camera_intrinsics is None:
                raise ValueError(
                    "camera conditioning requires camera_w2cs and "
                    "camera_intrinsics"
                )
            camera_w2cs, camera_intrinsics = (
                self._prepare_camera_conditioning(
                    camera_w2cs,
                    camera_intrinsics,
                    B,
                    num_views,
                    x.device,
                )
            )
            sample_batch = B // num_views
            if self.training and self.camera_conditioning_dropout > 0:
                camera_keep = (
                    torch.rand(sample_batch, 1, 1, device=x.device)
                    >= self.camera_conditioning_dropout
                ).to(x.dtype)
            else:
                camera_keep = torch.ones(
                    sample_batch, 1, 1, device=x.device, dtype=x.dtype
                )
            if self.camera_rays_enabled or self.pit_camera_rays_enabled:
                camera_ray_patches = build_plucker_ray_patches(
                    camera_w2cs,
                    camera_intrinsics,
                    H,
                    W,
                    self.patch_size,
                    x.dtype,
                )
                camera_ray_keep = camera_keep[:, None].expand(
                    sample_batch, num_views, 1, 1
                )
                if self.camera_ray_rows == "render":
                    row_mask = camera_ray_keep.new_zeros(
                        sample_batch, num_views, 1, 1
                    )
                    row_mask[:, 0] = 1
                    camera_ray_keep = camera_ray_keep * row_mask
                camera_ray_keep = camera_ray_keep.reshape(B, 1, 1)
            else:
                camera_ray_patches = None
                camera_ray_keep = None
        else:
            if camera_w2cs is not None or camera_intrinsics is not None:
                raise ValueError(
                    "camera tensors were supplied while "
                    "camera_conditioning_mode='none'"
                )
            camera_w2cs = camera_intrinsics = None
            camera_keep = None
            camera_ray_patches = camera_ray_keep = None

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
            for i in range(self.patch_depth):
                s, y_emb = self.patch_blocks[i](
                    s,
                    y_emb,
                    condition,
                    pos,
                    pos_txt,
                    attn_mask_joint,
                    num_views,
                    camera_ray_patches=camera_ray_patches,
                    camera_ray_keep=camera_ray_keep,
                    prope_w2cs=camera_w2cs,
                    prope_intrinsics=camera_intrinsics,
                    patch_grid=(Hs, Ws),
                    camera_keep=camera_keep,
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
        if depth is not None:
            x_pixels = x_pixels + self._embed_pixel_depth(depth, H, W)
        for blk in self.pixel_blocks:
            x_pixels = blk(
                x_pixels,
                s_cond,
                H,
                W,
                self.patch_size,
                mask,
                num_views=num_views,
                camera_ray_patches=camera_ray_patches,
                camera_ray_keep=camera_ray_keep,
                prope_w2cs=camera_w2cs,
                prope_intrinsics=camera_intrinsics,
                camera_keep=camera_keep,
            )

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
