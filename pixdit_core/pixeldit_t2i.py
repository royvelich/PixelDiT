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


def build_epipolar_attention_bias(
    w2cs: torch.Tensor,
    intrinsics: torch.Tensor,
    patch_grid,
    margin_patches: float,
    clip_value: float,
) -> torch.Tensor:
    """Return a depth-free cross-view score bias in ``[B,V,N,V*N]`` layout.

    The cameras use PRoPE's normalized image convention: both image axes cover
    approximately ``[-0.5, 0.5]`` and intrinsics include the shared resize/crop.
    Same-view entries stay zero so ordinary pretrained 2D-RoPE self-attention is
    untouched. Cross-view entries are the negative robust distance from a key
    patch centre to the query patch's epipolar line.
    """
    if patch_grid is None or len(patch_grid) != 2:
        raise ValueError("epipolar attention requires a (height, width) patch grid")
    patch_height, patch_width = (int(value) for value in patch_grid)
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"invalid epipolar patch grid {patch_grid}")
    if w2cs.dim() != 4 or w2cs.shape[-2:] != (4, 4):
        raise ValueError(f"w2cs must be [B,V,4,4], got {tuple(w2cs.shape)}")
    if intrinsics.shape != w2cs.shape[:2] + (3, 3):
        raise ValueError(
            "intrinsics must match w2cs as [B,V,3,3], "
            f"got {tuple(intrinsics.shape)}"
        )

    device = w2cs.device
    batch, views = w2cs.shape[:2]
    tokens = patch_height * patch_width
    with torch.autocast(device_type=device.type, enabled=False):
        xs = (
            (torch.arange(patch_width, device=device, dtype=torch.float32) + 0.5)
            / patch_width
            - 0.5
        )
        ys = (
            (torch.arange(patch_height, device=device, dtype=torch.float32) + 0.5)
            / patch_height
            - 0.5
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        patch_points = torch.stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                torch.ones(tokens, device=device, dtype=torch.float32),
            ),
            dim=-1,
        )

        w2cs = w2cs.float()
        intrinsics = intrinsics.float()
        c2ws = invert_se3(w2cs)
        inverse_intrinsics = torch.linalg.inv(intrinsics)
        patch_scale = 0.5 * (
            1.0 / float(patch_width) + 1.0 / float(patch_height)
        )
        output = torch.zeros(
            batch,
            views,
            tokens,
            views * tokens,
            device=device,
            dtype=torch.float32,
        )

        for query_view in range(views):
            for key_view in range(views):
                if query_view == key_view:
                    continue
                relative = w2cs[:, key_view] @ c2ws[:, query_view]
                rotation = relative[:, :3, :3]
                translation = relative[:, :3, 3]
                tx, ty, tz = translation.unbind(dim=-1)
                skew = torch.zeros(
                    batch, 3, 3, device=device, dtype=torch.float32
                )
                skew[:, 0, 1] = -tz
                skew[:, 0, 2] = ty
                skew[:, 1, 0] = tz
                skew[:, 1, 2] = -tx
                skew[:, 2, 0] = -ty
                skew[:, 2, 1] = tx
                fundamental = (
                    inverse_intrinsics[:, key_view].transpose(-1, -2)
                    @ skew
                    @ rotation
                    @ inverse_intrinsics[:, query_view]
                )
                lines = torch.einsum(
                    "bij,nj->bni", fundamental, patch_points
                )
                numerator = torch.einsum(
                    "bni,mi->bnm", lines, patch_points
                ).abs()
                denominator = torch.linalg.vector_norm(
                    lines[..., :2], dim=-1
                ).clamp_min(1e-8)
                distance_patches = (
                    numerator / denominator.unsqueeze(-1) / patch_scale
                )
                excess = F.relu(distance_patches - float(margin_patches))
                penalty = torch.sqrt(1.0 + excess.square()) - 1.0
                penalty = penalty.clamp_max(float(clip_value))
                valid_baseline = (
                    torch.linalg.vector_norm(translation, dim=-1) > 1e-7
                ).view(batch, 1, 1)
                start = key_view * tokens
                output[
                    :, query_view, :, start : start + tokens
                ] = -penalty * valid_baseline
    return output


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
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        camera_pose_mode: str = "none",
        multi_ref_attention_mode: str = "joint",
        pose_side_dim: int = 16,
        prope_freq_base: float = 100.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.camera_pose_mode = str(camera_pose_mode)
        self.multi_ref_attention_mode = str(multi_ref_attention_mode)
        self.pose_side_dim = int(pose_side_dim)
        if self.camera_pose_mode not in (
            "none",
            "joint_prope",
            "epipolar_bias",
            "prope_side",
        ):
            raise ValueError(f"unsupported camera pose mode {self.camera_pose_mode!r}")
        if self.multi_ref_attention_mode not in ("joint", "rank0_residual"):
            raise ValueError(
                "multi_ref_attention_mode must be joint|rank0_residual"
            )

        self.qkv_x = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)

        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)

        self.proj_x = nn.Linear(self.dim, self.dim)
        self.proj_y = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

        self.joint_prope = None
        if self.camera_pose_mode == "joint_prope":
            self.joint_prope = PropeDotProductAttention(
                self.head_dim,
                freq_base=prope_freq_base,
            )

        self.pose_gate = None
        if self.camera_pose_mode == "epipolar_bias":
            initial_pose_gate = (
                1.0
                if self.multi_ref_attention_mode == "rank0_residual"
                else 0.0
            )
            self.pose_gate = nn.Parameter(
                torch.full((self.num_heads,), initial_pose_gate)
            )

        self.pose_q = self.pose_k = self.pose_side_prope = None
        if self.camera_pose_mode == "prope_side":
            if self.pose_side_dim < 4 or self.pose_side_dim % 4:
                raise ValueError("pose_side_dim must be >= 4 and divisible by 4")
            side_width = self.num_heads * self.pose_side_dim
            self.pose_q = nn.Linear(self.dim, side_width, bias=False)
            self.pose_k = nn.Linear(self.dim, side_width, bias=False)
            nn.init.zeros_(self.pose_q.weight)
            nn.init.normal_(self.pose_k.weight, std=0.02)
            self.pose_side_prope = PropeDotProductAttention(
                self.pose_side_dim,
                freq_base=prope_freq_base,
            )

        self.extra_ref_gate = None
        if self.multi_ref_attention_mode == "rank0_residual":
            self.extra_ref_gate = nn.Parameter(torch.zeros(self.num_heads))

    @staticmethod
    def _validate_pose_inputs(
        prope_w2cs,
        prope_intrinsics,
        batch,
        num_views,
        patch_grid,
    ):
        if prope_w2cs is None or prope_intrinsics is None:
            raise ValueError("camera pose attention requires cameras and intrinsics")
        if patch_grid is None or len(patch_grid) != 2:
            raise ValueError("camera pose attention requires a patch grid")
        if prope_w2cs.shape != (batch, num_views, 4, 4):
            raise ValueError(
                "camera matrices must be "
                f"{(batch, num_views, 4, 4)}, got {tuple(prope_w2cs.shape)}"
            )
        if prope_intrinsics.shape != (batch, num_views, 3, 3):
            raise ValueError(
                "camera intrinsics must be "
                f"{(batch, num_views, 3, 3)}, "
                f"got {tuple(prope_intrinsics.shape)}"
            )

    @staticmethod
    def _prepare_prope(module, w2cs, intrinsics, patch_grid, device):
        patch_height, patch_width = (int(value) for value in patch_grid)
        module.update_coeffs(patch_width, patch_height, device)
        module.prepare(w2cs.float(), intrinsics.float())

    def _joint_image_attention(
        self,
        qx,
        kx,
        vx,
        num_views,
        tokens_per_view,
        pose_bias=None,
        pose_q=None,
        pose_k=None,
    ):
        if self.camera_pose_mode not in ("epipolar_bias", "prope_side"):
            return F.scaled_dot_product_attention(
                qx, kx, vx, dropout_p=0.0
            )

        batch, heads, _, width = qx.shape
        q_views = qx.view(
            batch, heads, num_views, tokens_per_view, width
        )
        pose_q_views = None
        if pose_q is not None:
            pose_q_views = pose_q.view(
                batch,
                heads,
                num_views,
                tokens_per_view,
                self.pose_side_dim,
            )
        outputs = []
        for query_view in range(num_views):
            score_bias = None
            if self.camera_pose_mode == "epipolar_bias":
                if pose_bias is None:
                    raise ValueError("epipolar attention requires pose_bias")
                score_bias = (
                    pose_bias[:, query_view].unsqueeze(1)
                    * self.pose_gate.view(1, heads, 1, 1)
                ).to(qx.dtype)
            elif self.camera_pose_mode == "prope_side":
                score_bias = torch.einsum(
                    "bhnd,bhmd->bhnm",
                    pose_q_views[:, :, query_view],
                    pose_k,
                ) * (self.pose_side_dim ** -0.5)
            outputs.append(
                F.scaled_dot_product_attention(
                    q_views[:, :, query_view],
                    kx,
                    vx,
                    dropout_p=0.0,
                    attn_mask=score_bias,
                )
            )
        return torch.stack(outputs, dim=2).reshape(
            batch, heads, num_views * tokens_per_view, width
        )

    def _rank0_residual_attention(
        self,
        qx,
        kx,
        vx,
        num_views,
        tokens_per_view,
        pose_bias=None,
        pose_q=None,
        pose_k=None,
    ):
        if num_views < 3:
            raise ValueError(
                "rank0_residual needs render, rank0, and at least one extra ref"
            )
        batch, heads, _, width = qx.shape
        q_views = qx.view(
            batch, heads, num_views, tokens_per_view, width
        )
        k_views = kx.view(
            batch, heads, num_views, tokens_per_view, width
        )
        v_views = vx.view(
            batch, heads, num_views, tokens_per_view, width
        )

        base_q = q_views[:, :, :2].reshape(
            batch, heads, 2 * tokens_per_view, width
        )
        base_k = k_views[:, :, :2].reshape_as(base_q)
        base_v = v_views[:, :, :2].reshape_as(base_q)
        base_output = F.scaled_dot_product_attention(
            base_q, base_k, base_v, dropout_p=0.0
        ).view(batch, heads, 2, tokens_per_view, width)

        extra_views = num_views - 2
        extra_q = q_views[:, :, 2:].permute(0, 2, 1, 3, 4).reshape(
            batch * extra_views, heads, tokens_per_view, width
        )
        extra_k = k_views[:, :, 2:].permute(0, 2, 1, 3, 4).reshape_as(
            extra_q
        )
        extra_v = v_views[:, :, 2:].permute(0, 2, 1, 3, 4).reshape_as(
            extra_q
        )
        extra_output = F.scaled_dot_product_attention(
            extra_q, extra_k, extra_v, dropout_p=0.0
        ).view(
            batch, extra_views, heads, tokens_per_view, width
        ).permute(0, 2, 1, 3, 4)

        cross_k = k_views[:, :, 2:].reshape(
            batch, heads, extra_views * tokens_per_view, width
        )
        cross_v = v_views[:, :, 2:].reshape_as(cross_k)
        cross_bias = None
        if self.camera_pose_mode == "epipolar_bias":
            if pose_bias is None:
                raise ValueError("epipolar attention requires pose_bias")
            cross_bias = (
                pose_bias[:, 0, :, 2 * tokens_per_view :].unsqueeze(1)
                * self.pose_gate.view(1, heads, 1, 1)
            ).to(qx.dtype)
        elif self.camera_pose_mode == "prope_side":
            pose_q_views = pose_q.view(
                batch,
                heads,
                num_views,
                tokens_per_view,
                self.pose_side_dim,
            )
            cross_pose_k = pose_k.view(
                batch,
                heads,
                num_views,
                tokens_per_view,
                self.pose_side_dim,
            )[:, :, 2:].reshape(
                batch,
                heads,
                extra_views * tokens_per_view,
                self.pose_side_dim,
            )
            cross_bias = torch.einsum(
                "bhnd,bhmd->bhnm",
                pose_q_views[:, :, 0],
                cross_pose_k,
            ) * (self.pose_side_dim ** -0.5)
        cross_output = F.scaled_dot_product_attention(
            q_views[:, :, 0],
            cross_k,
            cross_v,
            dropout_p=0.0,
            attn_mask=cross_bias,
        )
        render_output = (
            base_output[:, :, 0]
            + torch.tanh(self.extra_ref_gate).view(1, heads, 1, 1)
            * cross_output
        )
        output_views = torch.cat(
            (
                render_output.unsqueeze(2),
                base_output[:, :, 1:],
                extra_output,
            ),
            dim=2,
        )
        return output_views.reshape(
            batch, heads, num_views * tokens_per_view, width
        )

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
        pose_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image rows are b-major [render, ref0, ref1, ...]. New pose modes are
        # deliberately image-only so their score layouts cannot silently collide
        # with the shared text token stream.
        model_rows, tokens_per_view, channels = x.shape
        text_on = y is not None
        sample_batch = model_rows // num_views
        if text_on:
            sample_batch, text_tokens, text_channels = y.shape
            assert channels == text_channels
            assert model_rows == sample_batch * num_views
        if (
            text_on
            and (
                self.camera_pose_mode != "none"
                or self.multi_ref_attention_mode != "joint"
            )
        ):
            raise ValueError("pose and rank0-residual attention require no text")

        qkv_x = self.qkv_x(x).reshape(
            model_rows,
            tokens_per_view,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        pose_q = pose_k = None
        if self.camera_pose_mode == "prope_side":
            pose_q = self.pose_q(x).reshape(
                model_rows,
                tokens_per_view,
                self.num_heads,
                self.pose_side_dim,
            ).transpose(1, 2)
            pose_k = self.pose_k(x).reshape(
                model_rows,
                tokens_per_view,
                self.num_heads,
                self.pose_side_dim,
            ).transpose(1, 2)

        if text_on:
            qkv_y = self.qkv_y(y).reshape(
                sample_batch,
                text_tokens,
                3,
                self.num_heads,
                self.head_dim,
            ).permute(2, 0, 1, 3, 4)
            qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
            qy = self.q_norm_y(qy)
            ky = self.k_norm_y(ky)

        # The safe modes always retain the pretrained 2D RoPE. Only the
        # historical joint_prope control takes the old destructive switch.
        if self.joint_prope is None:
            qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if text_on and pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = merge_views(qx.transpose(1, 2), num_views)
        kx = merge_views(kx.transpose(1, 2), num_views)
        vx = merge_views(vx.transpose(1, 2), num_views)
        if pose_q is not None:
            pose_q = merge_views(pose_q, num_views)
            pose_k = merge_views(pose_k, num_views)

        if self.camera_pose_mode != "none":
            self._validate_pose_inputs(
                prope_w2cs,
                prope_intrinsics,
                sample_batch,
                num_views,
                patch_grid,
            )
        if self.joint_prope is not None:
            self._prepare_prope(
                self.joint_prope,
                prope_w2cs,
                prope_intrinsics,
                patch_grid,
                qx.device,
            )
            query_dtype = qx.dtype
            qx = self.joint_prope.apply_to_q(qx.float()).to(query_dtype)
            kx = self.joint_prope.apply_to_kv(kx.float()).to(query_dtype)
            vx = self.joint_prope.apply_to_kv(vx.float()).to(query_dtype)
        elif self.pose_side_prope is not None:
            self._prepare_prope(
                self.pose_side_prope,
                prope_w2cs,
                prope_intrinsics,
                patch_grid,
                qx.device,
            )
            side_dtype = pose_q.dtype
            pose_q = self.pose_side_prope.apply_to_q(
                pose_q.float()
            ).to(side_dtype)
            pose_k = self.pose_side_prope.apply_to_kv(
                pose_k.float()
            ).to(side_dtype)

        if text_on:
            qy = qy.transpose(1, 2)
            ky = ky.transpose(1, 2)
            vy = vy.transpose(1, 2)
            q_joint = torch.cat([qy, qx], dim=2)
            k_joint = torch.cat([ky, kx], dim=2)
            v_joint = torch.cat([vy, vx], dim=2)
            out_joint = F.scaled_dot_product_attention(
                q_joint,
                k_joint,
                v_joint,
                dropout_p=0.0,
                attn_mask=attn_mask,
            )
            image_output = out_joint[:, :, text_tokens:]
        elif self.multi_ref_attention_mode == "rank0_residual":
            image_output = self._rank0_residual_attention(
                qx,
                kx,
                vx,
                num_views,
                tokens_per_view,
                pose_bias=pose_bias,
                pose_q=pose_q,
                pose_k=pose_k,
            )
            out_joint = image_output
        else:
            image_output = self._joint_image_attention(
                qx,
                kx,
                vx,
                num_views,
                tokens_per_view,
                pose_bias=pose_bias,
                pose_q=pose_q,
                pose_k=pose_k,
            )
            out_joint = image_output

        if self.joint_prope is not None:
            output_dtype = image_output.dtype
            image_output = self.joint_prope.apply_to_output(
                image_output.float()
            ).to(output_dtype)

        out_x = split_views(
            image_output, num_views, tokens_per_view
        ).transpose(1, 2).reshape(
            model_rows, tokens_per_view, channels
        )
        out_x = self.proj_drop_x(self.proj_x(out_x))
        if not text_on:
            return out_x, None

        out_y = out_joint[:, :, :text_tokens].transpose(1, 2).reshape(
            sample_batch, text_tokens, channels
        )
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(
        self,
        hidden_size,
        groups,
        mlp_ratio=4.0,
        adaLN_modulation_img=None,
        adaLN_modulation_txt=None,
        camera_pose_mode="none",
        multi_ref_attention_mode="joint",
        pose_side_dim=16,
        prope_freq_base=100.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups

        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)

        self.attn = MMDiTJointAttention(
            hidden_size,
            num_heads=groups,
            qkv_bias=False,
            camera_pose_mode=camera_pose_mode,
            multi_ref_attention_mode=multi_ref_attention_mode,
            pose_side_dim=pose_side_dim,
            prope_freq_base=prope_freq_base,
        )

        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation_img = adaLN_modulation_img if adaLN_modulation_img is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = adaLN_modulation_txt if adaLN_modulation_txt is not None else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(
        self,
        x,
        y,
        c,
        pos_img,
        pos_txt=None,
        attn_mask=None,
        num_views=1,
        prope_w2cs=None,
        prope_intrinsics=None,
        patch_grid=None,
        pose_bias=None,
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
            pose_bias=pose_bias,
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
        camera_pose_mode: str = "none",
        multi_ref_attention_mode: str = "joint",
        pose_shuffle: bool = False,
        pose_bias_margin: float = 1.5,
        pose_bias_clip: float = 16.0,
        pose_side_dim: int = 16,
        prope_freq_base: float = 100.0,
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
        self.camera_pose_mode = str(camera_pose_mode)
        if self.camera_pose_mode not in (
            "none",
            "joint_prope",
            "epipolar_bias",
            "prope_side",
        ):
            raise ValueError(
                "camera_pose_mode must be "
                "none|joint_prope|epipolar_bias|prope_side, "
                f"got {self.camera_pose_mode!r}"
            )
        self.multi_ref_attention_mode = str(multi_ref_attention_mode)
        if self.multi_ref_attention_mode not in ("joint", "rank0_residual"):
            raise ValueError(
                "multi_ref_attention_mode must be joint|rank0_residual"
            )
        self.pose_shuffle = bool(pose_shuffle)
        self.pose_bias_margin = float(pose_bias_margin)
        self.pose_bias_clip = float(pose_bias_clip)
        self.pose_side_dim = int(pose_side_dim)
        self.joint_prope_enabled = self.camera_pose_mode == "joint_prope"
        self.camera_pose_enabled = self.camera_pose_mode != "none"
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
                camera_pose_mode=self.camera_pose_mode,
                multi_ref_attention_mode=self.multi_ref_attention_mode,
                pose_side_dim=self.pose_side_dim,
                prope_freq_base=prope_freq_base,
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

    @staticmethod
    def _prepare_pose_cameras(
        w2cs: torch.Tensor,
        intrinsics: torch.Tensor,
        model_rows: int,
        num_views: int,
        device,
        shuffle_references: bool = False,
    ):
        if w2cs.dim() == 3:
            w2cs = w2cs.unsqueeze(0)
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(0)
        if w2cs.dim() != 4 or w2cs.shape[1:] != (num_views, 4, 4):
            raise ValueError(
                "camera view matrices must be [B,V,4,4], "
                f"got {tuple(w2cs.shape)}"
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
        if model_batch % w2cs.shape[0]:
            raise ValueError(
                f"camera batch {w2cs.shape[0]} does not divide "
                f"model batch {model_batch}"
            )
        if intrinsics.shape[0] != w2cs.shape[0]:
            raise ValueError("PRoPE extrinsic and intrinsic batches disagree")
        if w2cs.shape[0] != model_batch:
            repeats = model_batch // w2cs.shape[0]
            w2cs = w2cs.repeat(repeats, 1, 1, 1)
            intrinsics = intrinsics.repeat(repeats, 1, 1, 1)
        if shuffle_references:
            if num_views < 3:
                raise ValueError(
                    "reference-pose shuffle requires at least two references"
                )
            # Keep the render camera fixed and cyclically attach every reference
            # image to a different reference camera. Deterministic in training
            # and validation, so this is a reproducible causality control.
            w2cs = torch.cat(
                (w2cs[:, :1], torch.roll(w2cs[:, 1:], 1, dims=1)), dim=1
            )
            intrinsics = torch.cat(
                (
                    intrinsics[:, :1],
                    torch.roll(intrinsics[:, 1:], 1, dims=1),
                ),
                dim=1,
            )
        return (
            w2cs.to(device=device, dtype=torch.float32),
            intrinsics.to(device=device, dtype=torch.float32),
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
        prope_w2cs=None,
        prope_intrinsics=None,
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
        if (
            text_on
            and (
                self.camera_pose_enabled
                or self.multi_ref_attention_mode != "joint"
            )
        ):
            raise ValueError(
                "pose and rank0-residual attention require the text stream "
                "to be disabled"
            )
        pose_bias = None
        if self.camera_pose_enabled:
            if prope_w2cs is None or prope_intrinsics is None:
                raise ValueError(
                    "camera pose attention requires prope_w2cs and "
                    "prope_intrinsics"
                )
            prope_w2cs, prope_intrinsics = self._prepare_pose_cameras(
                prope_w2cs,
                prope_intrinsics,
                B,
                num_views,
                x.device,
                shuffle_references=self.pose_shuffle,
            )
            if self.camera_pose_mode == "epipolar_bias":
                pose_bias = build_epipolar_attention_bias(
                    prope_w2cs,
                    prope_intrinsics,
                    (Hs, Ws),
                    self.pose_bias_margin,
                    self.pose_bias_clip,
                )
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
                    prope_w2cs=prope_w2cs,
                    prope_intrinsics=prope_intrinsics,
                    patch_grid=(Hs, Ws),
                    pose_bias=pose_bias,
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
