import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from pixdit_core.modules import (
    ClassEmbedder,
    FinalLayer,
    FeedForward,
    MLP,
    RMSNorm,
    RotaryAttention,
    TimestepConditioner,
    apply_adaln,
    apply_rotary_emb,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    precompute_freqs_cis_2d,
    scaled_dot_product_attention,
)
from pixdit_core.prope import PropeDotProductAttention


class PatchTokenEmbedder(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class AugmentedDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RotaryAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        if height == width:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, height)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
            self._pos_cache[key] = pos
            return pos
        else:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
            self._pos_cache[key] = pos
            return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None):
        if inputs.dim() != 4:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B,C,H,W]")
        assert img_height is not None and img_width is not None and patch_size is not None
        B, C, H, W = inputs.shape
        assert H == img_height and W == img_width
        assert (H % patch_size == 0) and (W % patch_size == 0)
        Hs, Ws = H // patch_size, W // patch_size
        P2 = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)
            pos_full = pos_full.view(H, W, self.hidden_size_output)
            x = x + pos_full.unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * Hs * Ws, P2, self.hidden_size_output)
        return x


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_fn=None,
        adaln_post_modulation: bool = False,
        view_attention: str = "separate",
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        assert (
            self.attn_dim % self.num_heads == 0
        ), "pixel attention hidden size must be divisible by pixel num_heads"
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaln_post_modulation = bool(adaln_post_modulation)
        self.view_attention = str(view_attention)
        if self.view_attention not in ("separate", "joint"):
            raise ValueError("PiT view_attention must be separate|joint")
        n_mod = 4 if self.adaln_post_modulation else 6
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, n_mod * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d
        self.pit_ray_embedder = None
        self.pit_prope_ref_k = None
        self.pit_prope_ref_v = None
        self.pit_prope_ref_k_norm = None
        self.pit_prope_source = None
        self.pit_prope_references = None

    def enable_camera_conditioning(
        self,
        rays_enabled: bool,
        prope_enabled: bool,
        prope_freq_base: float = 100.0,
    ) -> None:
        """Add optional camera conditioning directly to the PiT attention path.

        Rays are projected per pixel before the patch's ``P*P`` pixel features are
        compressed into one attention token. PRoPE is an additional asymmetric
        render-query/reference-KV residual, matching the patch-level implementation.
        Every new output path is zero-initialized.
        """
        if rays_enabled and self.pit_ray_embedder is None:
            self.pit_ray_embedder = nn.Linear(6, self.pixel_dim, bias=True)
            nn.init.zeros_(self.pit_ray_embedder.weight)
            nn.init.zeros_(self.pit_ray_embedder.bias)
        if prope_enabled and self.pit_prope_ref_k is None:
            self.pit_prope_ref_k = nn.Linear(
                self.attn_dim, self.attn_dim, bias=True
            )
            self.pit_prope_ref_v = nn.Linear(
                self.attn_dim, self.attn_dim, bias=True
            )
            self.pit_prope_ref_k_norm = RMSNorm(self.attn_dim // self.num_heads)
            self.pit_prope_source = PropeDotProductAttention(
                self.attn_dim // self.num_heads,
                freq_base=prope_freq_base,
            )
            self.pit_prope_references = PropeDotProductAttention(
                self.attn_dim // self.num_heads,
                freq_base=prope_freq_base,
            )
            nn.init.zeros_(self.pit_prope_ref_v.weight)
            nn.init.zeros_(self.pit_prope_ref_v.bias)

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = self._rope_fn(self.attn_dim // self.num_heads, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    @staticmethod
    def _merge_views(features: torch.Tensor, num_views: int) -> torch.Tensor:
        """``[B*V,H,L,D]`` -> ``[B,H,V*L,D]`` for joint PiT attention."""
        if num_views == 1:
            return features
        batch_rows, heads, tokens, width = features.shape
        if batch_rows % num_views:
            raise ValueError(
                f"PiT rows {batch_rows} are not divisible by num_views={num_views}"
            )
        batch = batch_rows // num_views
        return (
            features.view(batch, num_views, heads, tokens, width)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, heads, num_views * tokens, width)
        )

    @staticmethod
    def _split_views(
        features: torch.Tensor,
        num_views: int,
        tokens_per_view: int,
    ) -> torch.Tensor:
        """Exact inverse of :meth:`_merge_views`."""
        if num_views == 1:
            return features
        batch, heads, all_tokens, width = features.shape
        if all_tokens != num_views * tokens_per_view:
            raise ValueError(
                f"PiT token axis {all_tokens} != "
                f"num_views {num_views} * tokens {tokens_per_view}"
            )
        return (
            features.view(
                batch, heads, num_views, tokens_per_view, width
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(
                batch * num_views, heads, tokens_per_view, width
            )
        )

    def _prope_reference_attention(
        self,
        x: torch.Tensor,
        render_query: torch.Tensor,
        num_views: int,
        viewmats: torch.Tensor,
        intrinsics: torch.Tensor,
        patch_grid,
        camera_keep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return a PRoPE residual on render rows; reference rows stay zero."""
        if num_views <= 1:
            return torch.zeros_like(x)
        batch_rows, tokens, channels = x.shape
        if batch_rows % num_views:
            raise ValueError(
                f"PiT rows {batch_rows} are not divisible by num_views={num_views}"
            )
        batch = batch_rows // num_views
        if viewmats.shape != (batch, num_views, 4, 4):
            raise ValueError(
                "PiT PRoPE view matrices must be "
                f"{(batch, num_views, 4, 4)}, got {tuple(viewmats.shape)}"
            )
        if intrinsics.shape != (batch, num_views, 3, 3):
            raise ValueError(
                "PiT PRoPE intrinsics must be "
                f"{(batch, num_views, 3, 3)}, got {tuple(intrinsics.shape)}"
            )
        if patch_grid is None or len(patch_grid) != 2:
            raise ValueError("PiT PRoPE requires a patch grid")
        patch_height, patch_width = (int(value) for value in patch_grid)
        if patch_height * patch_width != tokens:
            raise ValueError(
                f"PiT patch grid {(patch_height, patch_width)} has "
                f"{patch_height * patch_width} tokens, expected {tokens}"
            )

        num_refs = num_views - 1
        references = x.view(batch, num_views, tokens, channels)[:, 1:]
        key = self.pit_prope_ref_k(references).view(
            batch,
            num_refs,
            tokens,
            self.num_heads,
            self.attn_dim // self.num_heads,
        )
        value = self.pit_prope_ref_v(references).view(
            batch,
            num_refs,
            tokens,
            self.num_heads,
            self.attn_dim // self.num_heads,
        )
        key = self.pit_prope_ref_k_norm(key)
        key = key.permute(0, 3, 1, 2, 4).reshape(
            batch,
            self.num_heads,
            num_refs * tokens,
            self.attn_dim // self.num_heads,
        )
        value = value.permute(0, 3, 1, 2, 4).reshape_as(key)
        query = (
            render_query.view(
                batch,
                num_views,
                tokens,
                self.num_heads,
                self.attn_dim // self.num_heads,
            )[:, 0]
            .permute(0, 2, 1, 3)
        )

        self.pit_prope_source.update_coeffs(
            patch_width, patch_height, query.device
        )
        self.pit_prope_references.update_coeffs(
            patch_width, patch_height, query.device
        )
        self.pit_prope_source.prepare(viewmats[:, :1], intrinsics[:, :1])
        self.pit_prope_references.prepare(
            viewmats[:, 1:], intrinsics[:, 1:]
        )

        attention_dtype = query.dtype
        query = self.pit_prope_source.apply_to_q(query).to(attention_dtype)
        key = self.pit_prope_references.apply_to_kv(key).to(attention_dtype)
        value = self.pit_prope_references.apply_to_kv(value).to(
            attention_dtype
        )
        output = scaled_dot_product_attention(
            query, key, value, dropout_p=0.0
        )
        output = self.pit_prope_source.apply_to_output(output).to(
            attention_dtype
        )
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        if camera_keep is not None:
            if camera_keep.shape != (batch, 1, 1):
                raise ValueError(
                    "PiT camera_keep must be "
                    f"{(batch, 1, 1)}, got {tuple(camera_keep.shape)}"
                )
            output = output * camera_keep.to(output.dtype)
        zeros = output.new_zeros(batch, num_refs, tokens, channels)
        return torch.cat([output[:, None], zeros], dim=1).reshape_as(x)

    def _attention(
        self,
        x: torch.Tensor,
        pos,
        mask,
        num_views: int,
        prope_w2cs: Optional[torch.Tensor],
        prope_intrinsics: Optional[torch.Tensor],
        patch_grid,
        camera_keep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the original per-view or the optional joint-view PiT attention."""
        batch_rows, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        qkv = (
            self.attn.qkv(x)
            .reshape(
                batch_rows, tokens, 3, self.num_heads, head_dim
            )
            .permute(2, 0, 1, 3, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.attn.q_norm(query)
        key = self.attn.k_norm(key)
        render_query = query
        query, key = apply_rotary_emb(query, key, freqs_cis=pos)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if self.view_attention == "joint" and num_views > 1:
            if mask is not None:
                raise ValueError(
                    "joint PiT attention does not support a per-view mask"
                )
            query = self._merge_views(query, num_views)
            key = self._merge_views(key, num_views)
            value = self._merge_views(value, num_views)
        output = scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0
        )
        if self.view_attention == "joint" and num_views > 1:
            output = self._split_views(output, num_views, tokens)
        output = output.transpose(1, 2).reshape(
            batch_rows, tokens, channels
        )

        if self.pit_prope_ref_k is not None:
            if prope_w2cs is None or prope_intrinsics is None:
                raise ValueError(
                    "PiT PRoPE reference attention requires camera matrices"
                )
            output = output + self._prope_reference_attention(
                x,
                render_query,
                num_views,
                prope_w2cs,
                prope_intrinsics,
                patch_grid,
                camera_keep,
            )
        output = self.attn.proj(output)
        return self.attn.proj_drop(output)

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        image_height: int,
        image_width: int,
        patch_size: int,
        mask=None,
        num_views: int = 1,
        camera_ray_patches: Optional[torch.Tensor] = None,
        camera_ray_keep: Optional[torch.Tensor] = None,
        prope_w2cs: Optional[torch.Tensor] = None,
        prope_intrinsics: Optional[torch.Tensor] = None,
        camera_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        BL, P2, C = x.shape
        if C != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0)
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        B = BL // L
        if BL != B * L:
            raise ValueError(f"PiT pixel rows {BL} are not divisible by L={L}")
        if B % num_views:
            raise ValueError(
                f"PiT image rows {B} are not divisible by num_views={num_views}"
            )
        n_mod = 4 if self.adaln_post_modulation else 6
        cond_params = self.adaLN_modulation(s_cond).view(BL, P2, n_mod * self.pixel_dim)
        if self.adaln_post_modulation:
            scale1, shift1, scale2, shift2 = torch.chunk(cond_params, 4, dim=-1)
            x_norm = self.norm1(x)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
            x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        if self.pit_ray_embedder is not None:
            if camera_ray_patches is None:
                raise ValueError(
                    "PiT ray conditioning requires camera_ray_patches"
                )
            expected = (B, L, 6 * P2)
            if camera_ray_patches.shape != expected:
                raise ValueError(
                    "PiT camera ray patches must be "
                    f"{expected}, got {tuple(camera_ray_patches.shape)}"
                )
            ray_pixels = (
                camera_ray_patches.view(B, L, 6, P2)
                .permute(0, 1, 3, 2)
                .reshape(BL, P2, 6)
            )
            ray_embedding = self.pit_ray_embedder(ray_pixels)
            if camera_ray_keep is not None:
                if camera_ray_keep.shape != (B, 1, 1):
                    raise ValueError(
                        "PiT camera_ray_keep must be "
                        f"{(B, 1, 1)}, got {tuple(camera_ray_keep.shape)}"
                    )
                ray_embedding = ray_embedding * camera_ray_keep.repeat_interleave(
                    L, dim=0
                ).to(ray_embedding.dtype)
            x_norm = x_norm + ray_embedding
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L, self.attn_dim)
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        attn_out = self._attention(
            x_comp,
            pos_comp,
            mask,
            num_views,
            prope_w2cs,
            prope_intrinsics,
            (Hs, Ws),
            camera_keep,
        )
        attn_flat = self.expand_from_attn(attn_out.view(B * L, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        if self.adaln_post_modulation:
            x = x + attn_exp * (1 + scale1) + shift1
            mlp_out = self.mlp(self.norm2(x))
            x = x + mlp_out * (1 + scale2) + shift2
        else:
            x = x + gate_msa * attn_exp
            mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
            x = x + gate_mlp * mlp_out
        return x


class PixDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_groups=12,
        hidden_size=1152,
        pixel_hidden_size=64,
        patch_depth=18,
        pixel_depth=4,
        patch_size=2,
        num_classes=1000,
        use_pixel_abs_pos=True,
        pit_adaln_post_modulation=False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT expects pixel_depth > 0 to preserve the dual-level pipeline")

        self.pixel_embedder = PixelTokenEmbedder(self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size ** 2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_size)

        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)
        self.patch_blocks = nn.ModuleList(
            [AugmentedDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.patch_depth)]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        for block in self.patch_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)
        for block in self.pixel_blocks:
            nn.init.zeros_(block.adaLN_modulation[0].weight)
            nn.init.zeros_(block.adaLN_modulation[0].bias)

    def forward(self, x, t, y, s=None, mask=None):
        B, _, H, W = x.shape
        pos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y_emb = self.y_embedder(y).view(B, 1, self.hidden_size)
        c = nn.functional.silu(t_emb + y_emb)
        if s is None:
            s = self.s_embedder(x_patches)
            for block in self.patch_blocks:
                s = block(s, c, pos, mask)
            s = nn.functional.silu(t_emb + s)
        batch_size, length, _ = s.shape
        s_cond = s.view(batch_size * length, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)
        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, length, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, length)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img
