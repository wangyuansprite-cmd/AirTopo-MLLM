"""Query-conditioned residual modulation for the SegVol dense G4 grid.

The MLLM vision path is deliberately untouched.  z_SEG is first produced from
the original visual tokens, then conditions only the dense SegVol image feature
immediately before the existing coarse mask transformer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .high_resolution_decoder import CTSkipMaskDecoder


class QueryResidualModulation(nn.Module):
    """Identity-at-step-zero residual channel gate or residual FiLM."""

    def __init__(self, z_dim: int, channels: int, mode: str):
        super().__init__()
        if mode not in {"baseline", "residual_gate", "residual_film"}:
            raise ValueError(mode)
        self.z_dim = int(z_dim)
        self.channels = int(channels)
        self.mode = mode
        self.runtime = {}
        # In bf16, a sufficiently small multiplicative residual is rounded out
        # by 1+x. This fixed reparameterization preserves alpha=0 identity while
        # making the learned effective scalar representable in the frozen path.
        self.gate_alpha_scale = 32.0
        if mode == "residual_gate":
            self.gate_mlp = nn.Sequential(
                nn.Linear(z_dim, channels), nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.alpha_gate = nn.Parameter(torch.zeros(()))
        else:
            self.gate_mlp = None
            self.register_parameter("alpha_gate", None)
        if mode == "residual_film":
            self.gamma_mlp = nn.Sequential(
                nn.Linear(z_dim, channels), nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.beta_mlp = nn.Sequential(
                nn.Linear(z_dim, channels), nn.GELU(),
                nn.Linear(channels, channels),
            )
            self.alpha_film = nn.Parameter(torch.zeros(()))
            # alpha=0 gives exact baseline equivalence.  A small deterministic
            # final projection is necessary so alpha receives a first-step
            # gradient; alpha=0 plus zero final layers would be a dead branch.
            nn.init.normal_(self.gamma_mlp[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.gamma_mlp[-1].bias)
            nn.init.normal_(self.beta_mlp[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.beta_mlp[-1].bias)
        else:
            self.gamma_mlp = None
            self.beta_mlp = None
            self.register_parameter("alpha_film", None)

    @staticmethod
    def _stats(value: torch.Tensor | None) -> dict:
        if value is None:
            return {"mean": None, "std": None, "min": None, "max": None}
        detached = value.detach().float()
        return {
            "mean": float(detached.mean()),
            "std": float(detached.std(unbiased=False)),
            "min": float(detached.min()),
            "max": float(detached.max()),
        }

    def forward(self, feature: torch.Tensor, z_seg: torch.Tensor):
        if feature.ndim != 5 or feature.shape[1] != self.channels:
            raise RuntimeError(f"unexpected dense feature shape {feature.shape}")
        if z_seg.ndim != 2 or z_seg.shape[1] != self.z_dim:
            raise RuntimeError(f"unexpected z_SEG shape {z_seg.shape}")
        gate = gamma = beta = None
        if self.mode == "baseline":
            output = feature
            alpha = None
        elif self.mode == "residual_gate":
            gate = torch.tanh(self.gate_mlp(z_seg))
            alpha = self.gate_alpha_scale * self.alpha_gate
            output = feature * (
                1.0 + alpha * gate[:, :, None, None, None]
            )
        else:
            gamma = self.gamma_mlp(z_seg)
            beta = self.beta_mlp(z_seg)
            alpha = self.alpha_film
            residual = (
                gamma[:, :, None, None, None] * feature
                + beta[:, :, None, None, None]
            )
            output = feature + alpha * residual
        with torch.no_grad():
            base_norm = torch.linalg.vector_norm(feature.detach().float())
            delta_norm = torch.linalg.vector_norm(
                output.detach().float() - feature.detach().float()
            )
            self.runtime = {
                "mode": self.mode,
                "z_seg_shape": list(z_seg.shape),
                "g4_shape": list(feature.shape),
                "g4_hat_shape": list(output.shape),
                "feature_max_abs_difference": float(
                    torch.max(torch.abs(output.detach().float() - feature.detach().float()))
                ),
                "delta_g_relative_norm": float(delta_norm / base_norm.clamp_min(1e-12)),
                "alpha": None if alpha is None else float(alpha.detach().float()),
                "alpha_raw": (
                    None if self.alpha_gate is None
                    else float(self.alpha_gate.detach().float())
                ),
                "gate_stats": self._stats(gate),
                "gamma_stats": self._stats(gamma),
                "beta_stats": self._stats(beta),
                "gate_vector": None if gate is None else gate.detach().float().cpu(),
                "gamma_vector": None if gamma is None else gamma.detach().float().cpu(),
                "beta_vector": None if beta is None else beta.detach().float().cpu(),
            }
        return output


class QueryAwareCTSkipMaskDecoder(CTSkipMaskDecoder):
    """Apply query modulation before the unchanged coarse decoder."""

    def __init__(self, base_decoder, z_dim: int, mode: str):
        # Construct the shared CT stem first so its seeded weights are exactly
        # identical for A/B/C, regardless of modulation parameter count.
        super().__init__(base_decoder, "addition")
        self.modulation = QueryResidualModulation(z_dim, 768, mode)

    def predict_masks(
        self, image_embeddings, ct_image, text_embedding, image_pe,
        sparse_prompt_embeddings, dense_prompt_embeddings,
    ):
        modulated = self.modulation(image_embeddings, text_embedding)
        return super().predict_masks(
            image_embeddings=modulated,
            ct_image=ct_image,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )


class QueryAwareCTSkipSegVol(nn.Module):
    """SegVol wrapper preserving its prompt conditioning and CT-Skip path."""

    def __init__(self, base_segvol, z_dim: int, mode: str):
        super().__init__()
        self.image_encoder = base_segvol.image_encoder
        self.prompt_encoder = base_segvol.prompt_encoder
        self.feat_shape = np.asarray(base_segvol.feat_shape)
        self.mask_decoder = QueryAwareCTSkipMaskDecoder(
            base_segvol.mask_decoder, z_dim, mode
        )

    def forward(self, image, text_emb=None, text=None, boxes=None, points=None):
        batch = image.shape[0]
        image_shape = tuple(image.shape[-3:])
        image_embedding, _ = self.image_encoder(image)
        expected_tokens = int(np.prod(self.feat_shape))
        if image_embedding.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} image tokens, got {image_embedding.shape[1]}"
            )
        image_embedding = image_embedding.transpose(1, 2).view(
            batch, -1, *(int(v) for v in self.feat_shape)
        )
        sparse, dense = self.prompt_encoder(
            points=None, boxes=None, masks=None, text_embedding=text_emb
        )
        dense_position = self.prompt_encoder.get_dense_pe()
        low_resolution_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            ct_image=image,
            text_embedding=text_emb,
            image_pe=dense_position,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return F.interpolate(
            low_resolution_masks, size=image_shape,
            mode="trilinear", align_corners=False,
        )


def get_modulation(model) -> QueryResidualModulation:
    return model.get_model().seg_module.mask_decoder.modulation


def configure_query_aware_variant(model, variant: str) -> dict:
    modes = {
        "A_g4_baseline": "baseline",
        "B_residual_gate": "residual_gate",
        "C_residual_film": "residual_film",
    }
    if variant not in modes:
        raise ValueError(variant)
    core = model.get_model()
    before_ids = {id(parameter) for parameter in model.parameters()}
    before_total = sum(parameter.numel() for parameter in model.parameters())
    base_segvol = core.seg_module
    z_dim = int(getattr(model.config, "mm_hidden_size"))
    reference = next(base_segvol.parameters())
    # New-module initialization must not shift the global RNG stream used by
    # LoRA dropout during training.  Preserve both CPU and CUDA states so all
    # controlled variants start their optimizer schedule with the same stream.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        wrapper = QueryAwareCTSkipSegVol(base_segvol, z_dim, modes[variant])
        wrapper.to(device=reference.device, dtype=reference.dtype)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    core.seg_module = wrapper
    after_total = sum(parameter.numel() for parameter in model.parameters())
    new = [(name, p) for name, p in model.named_parameters() if id(p) not in before_ids]
    modulation_parameters = sum(
        parameter.numel() for name, parameter in new if ".modulation." in name
    )
    ct_skip_parameters = sum(
        parameter.numel() for name, parameter in new if "ct_skip_stem" in name
    )
    return {
        "variant": variant,
        "mode": modes[variant],
        "z_seg_dimension": z_dim,
        "g4_shape": ["B", 768, 32, 12, 12],
        "preexisting_parameters_reused": before_total,
        "total_parameters": after_total,
        "added_parameters": after_total - before_total,
        "query_modulation_parameters": modulation_parameters,
        "ct_skip_parameters": ct_skip_parameters,
        "random_initialized_parameters": after_total - before_total,
        "pretrained_reuse_ratio": before_total / after_total,
        "new_parameter_names": [name for name, _ in new],
        "visual_token_path_modified": False,
        "decoder_architecture_modified": False,
        "ct_skip_modified": False,
    }
