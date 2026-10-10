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


class InterleavedReferenceHeadAttention(nn.Module):
    """Fuse pairwise render updates across references with grouped attention.

    The caller runs one ordinary render/reference MMDiT block per reference and
    passes the resulting render-side residual updates as
    ``[B,K,L,D]``.  For every patch and head, the current shared render produces
    one query and every reference update produces one key.  Softmax is only
    across the K references: there is deliberately no null/no-reference route.

    Values are the unprojected head groups from the pairwise render updates.
    This preserves the pretrained block update at K=1 exactly and avoids
    introducing a second feature-generating path.  The key projection starts at
    zero, so K>1 initially reduces to a stable uniform mean while still
    receiving a non-zero first-step gradient through the non-zero query.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        score_dim: int = 32,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_heads <= 0 or dim % num_heads:
            raise ValueError(
                "interleaved reference attention requires dim divisible by "
                f"num_heads, got dim={dim}, num_heads={num_heads}"
            )
        if score_dim <= 0:
            raise ValueError("interleaved reference attention score_dim must be > 0")
        if temperature <= 0:
            raise ValueError(
                "interleaved reference attention temperature must be > 0"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.score_dim = int(score_dim)
        self.temperature = float(temperature)

        self.query_weight = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.score_dim)
        )
        self.key_weight = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim, self.score_dim)
        )
        nn.init.xavier_uniform_(self.query_weight)
        nn.init.zeros_(self.key_weight)

        # Diagnostics only; never enters the state dict.
        self.last_attention_weights = None

    @staticmethod
    def _rms_normalize(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True).add(1e-6)
        ).to(dtype=x.dtype)

    def forward(
        self,
        render: torch.Tensor,
        reference_updates: torch.Tensor,
    ) -> torch.Tensor:
        if render.dim() != 3:
            raise ValueError(
                f"render must be [B,L,D], got {tuple(render.shape)}"
            )
        if reference_updates.dim() != 4:
            raise ValueError(
                "reference_updates must be [B,K,L,D], got "
                f"{tuple(reference_updates.shape)}"
            )
        if reference_updates.shape[1] < 1:
            raise ValueError("at least one reference update is required")
        if (
            reference_updates.shape[0] != render.shape[0]
            or reference_updates.shape[2:] != render.shape[1:]
        ):
            raise ValueError(
                "render/reference update shapes are incompatible: "
                f"{tuple(render.shape)} vs {tuple(reference_updates.shape)}"
            )

        # K=1 must reproduce the original rank-0 branch exactly.  Bypassing all
        # projection arithmetic also makes old single-reference experiments a
        # trustworthy compatibility test for this mode.
        if reference_updates.shape[1] == 1:
            self.last_attention_weights = reference_updates.new_ones(
                reference_updates.shape[0],
                1,
                reference_updates.shape[2],
                self.num_heads,
            )
            return reference_updates[:, 0]

        batch, references, tokens, _ = reference_updates.shape
        render_heads = render.view(
            batch, tokens, self.num_heads, self.head_dim
        )
        update_heads = reference_updates.view(
            batch, references, tokens, self.num_heads, self.head_dim
        )
        render_heads = self._rms_normalize(render_heads)
        update_heads_norm = self._rms_normalize(update_heads)

        queries = torch.einsum(
            "blhd,hds->blhs",
            render_heads,
            self.query_weight,
        )
        keys = torch.einsum(
            "bklhd,hds->bklhs",
            update_heads_norm,
            self.key_weight,
        )
        logits = torch.einsum(
            "blhs,bklhs->bklh",
            queries,
            keys,
        )
        logits = logits * (self.score_dim ** -0.5) / self.temperature
        weights = torch.softmax(logits.float(), dim=1).to(
            dtype=reference_updates.dtype
        )
        fused_heads = (
            weights[..., None] * update_heads
        ).sum(dim=1)
        self.last_attention_weights = weights.detach()
        return fused_heads.reshape(batch, tokens, self.dim)


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
        interleaved_ref_attention_enabled: bool = False,
        interleaved_ref_attention_heads: int = 0,
        interleaved_ref_attention_score_dim: int = 32,
        interleaved_ref_attention_temperature: float = 1.0,
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
        self.interleaved_ref_attention_enabled = bool(
            interleaved_ref_attention_enabled
        )
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
        fusion_heads = (
            int(interleaved_ref_attention_heads)
            if int(interleaved_ref_attention_heads) > 0
            else self.num_groups
        )
        self.interleaved_ref_fusions = (
            nn.ModuleList(
                [
                    InterleavedReferenceHeadAttention(
                        self.hidden_size,
                        fusion_heads,
                        score_dim=int(interleaved_ref_attention_score_dim),
                        temperature=float(
                            interleaved_ref_attention_temperature
                        ),
                    )
                    for _ in range(self.patch_depth)
                ]
            )
            if self.interleaved_ref_attention_enabled
            else None
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

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
        self.last_repa_tokens = None

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
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

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
                    attention_views = (
                        2
                        if (
                            self.interleaved_ref_attention_enabled
                            and num_views > 1
                        )
                        else num_views
                    )
                    # The key axis is [text | V*L image tokens] at batch Btxt after the merge.
                    # Image tokens are never padded, so their half stays all-False.
                    pad_img = torch.zeros(
                        (Btxt, attention_views * L),
                        dtype=torch.bool,
                        device=x.device,
                    )
                    attn_mask_joint = torch.cat([pad[:, :Ltxt], pad_img], dim=1).view(
                        Btxt, 1, 1, Ltxt + attention_views * L
                    )
            self.last_repa_tokens = None
            s = s0
            for i in range(self.patch_depth):
                if (
                    self.interleaved_ref_attention_enabled
                    and num_views > 1
                ):
                    s, y_emb = self._interleaved_reference_block(
                        self.patch_blocks[i],
                        self.interleaved_ref_fusions[i],
                        s,
                        y_emb,
                        condition,
                        pos,
                        pos_txt,
                        attn_mask_joint,
                        num_views,
                    )
                else:
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

    @staticmethod
    def _interleaved_reference_block(
        block: MMDiTBlockT2I,
        fusion: InterleavedReferenceHeadAttention,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        condition: torch.Tensor,
        pos_img: torch.Tensor,
        pos_txt: torch.Tensor,
        attn_mask: torch.Tensor,
        num_views: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one shared-render/reference pair per reference, then fuse.

        Rows use the repository-wide b-major convention
        ``[sample0 render, sample0 ref0, ..., sample1 render, ...]``.
        Every pair sees the same incoming render state.  Reference states remain
        separate, while the K candidate render residual updates are fused into
        the one render row supplied to the next patch block.
        """
        if num_views < 2:
            raise ValueError(
                "interleaved reference attention requires num_views >= 2"
            )
        rows, tokens, dim = image_tokens.shape
        if rows % num_views:
            raise ValueError(
                f"image token rows {rows} are not divisible by num_views={num_views}"
            )
        if condition.shape[0] != rows:
            raise ValueError(
                f"condition rows {condition.shape[0]} do not match image rows {rows}"
            )

        batch = rows // num_views
        references = num_views - 1
        if references == 1:
            # Preserve the original rank-0 architecture bit for bit.  In
            # particular, avoid the otherwise harmless ``render + (candidate -
            # render)`` round trip, which can change the last bit in bf16/fp32.
            return block(
                image_tokens,
                text_tokens,
                condition,
                pos_img,
                pos_txt,
                attn_mask,
                num_views=2,
            )

        views = image_tokens.view(batch, num_views, tokens, dim)
        condition_views = condition.view(
            batch, num_views, *condition.shape[1:]
        )
        render = views[:, 0]
        render_condition = condition_views[:, 0]

        render_candidates = []
        reference_outputs = []
        text_candidates = []
        for reference_index in range(references):
            reference = views[:, reference_index + 1]
            reference_condition = condition_views[:, reference_index + 1]
            pair_tokens = torch.stack(
                [render, reference],
                dim=1,
            ).reshape(batch * 2, tokens, dim)
            pair_condition = torch.stack(
                [render_condition, reference_condition],
                dim=1,
            ).reshape(batch * 2, *condition.shape[1:])
            pair_output, pair_text = block(
                pair_tokens,
                text_tokens,
                pair_condition,
                pos_img,
                pos_txt,
                attn_mask,
                num_views=2,
            )
            pair_output = pair_output.view(batch, 2, tokens, dim)
            render_candidates.append(pair_output[:, 0])
            reference_outputs.append(pair_output[:, 1])
            if pair_text is not None:
                text_candidates.append(pair_text)

        candidate_updates = (
            torch.stack(render_candidates, dim=1)
            - render[:, None]
        )
        render_next = render + fusion(render, candidate_updates)
        references_next = torch.stack(reference_outputs, dim=1)
        image_next = torch.cat(
            [render_next[:, None], references_next],
            dim=1,
        ).reshape(rows, tokens, dim)

        text_next = None
        if text_candidates:
            # Text remains one shared stream per sample.  Averaging the K
            # pair-specific updates is permutation-invariant and exactly
            # recovers the original behavior when K=1.
            text_next = torch.stack(text_candidates, dim=0).mean(dim=0)
        return image_next, text_next
