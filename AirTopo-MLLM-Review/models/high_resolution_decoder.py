"""Isolated SegVol high-resolution CT skip branches and fusion wrappers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class HighResolutionCTStem(nn.Module):
    """Map native CT to a 96-channel 2x-downsampled XY feature."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True
        )
        self.norm1 = nn.GroupNorm(8, 32)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(
            32, 64, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm2 = nn.GroupNorm(8, 64)
        self.act2 = nn.GELU()
        self.project = nn.Conv3d(
            64, 96, kernel_size=1, stride=1, padding=0, bias=True
        )
        nn.init.kaiming_normal_(
            self.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.conv1.bias)
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.kaiming_normal_(
            self.conv2.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.conv2.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.project.bias)

    def forward(self, image):
        return self.project(
            self.act2(self.norm2(self.conv2(
                self.act1(self.norm1(self.conv1(image)))
            )))
        )


class CTSkipMaskDecoder(nn.Module):
    """Generate language-conditioned masks after high-resolution CT fusion."""

    def __init__(self, base_decoder: nn.Module, fusion: str):
        super().__init__()
        if fusion not in {"addition", "gated"}:
            raise ValueError(fusion)
        self.base_decoder = base_decoder
        self.fusion = fusion
        self.ct_skip_stem = HighResolutionCTStem()
        self.gate = (
            nn.Conv3d(192, 96, kernel_size=1, stride=1, bias=True)
            if fusion == "gated" else None
        )
        if self.gate is not None:
            nn.init.normal_(self.gate.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.gate.bias, -2.0)
        self.runtime_shapes = {}

    def _fuse(self, coarse_feature, ct_image):
        coarse_high = F.interpolate(
            coarse_feature,
            scale_factor=(1, 2, 2),
            mode="trilinear",
            align_corners=False,
        )
        skip = self.ct_skip_stem(ct_image)
        if coarse_high.shape != skip.shape:
            raise RuntimeError(
                f"coarse/skip shape mismatch: {coarse_high.shape}, {skip.shape}"
            )
        if self.fusion == "addition":
            fused = coarse_high + skip
            gate = None
        else:
            gate = torch.sigmoid(
                self.gate(torch.cat((coarse_high, skip), dim=1))
            )
            fused = coarse_high + gate * skip
        self.runtime_shapes = {
            "input_ct": list(ct_image.shape),
            "pretrained_coarse_feature": list(coarse_feature.shape),
            "upsampled_coarse_feature": list(coarse_high.shape),
            "ct_skip_feature": list(skip.shape),
            "gate": list(gate.shape) if gate is not None else None,
            "fused_feature": list(fused.shape),
        }
        return fused

    def forward(
        self,
        image_embeddings: torch.Tensor,
        ct_image: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        masks, iou_prediction = self.predict_masks(
            image_embeddings=image_embeddings,
            ct_image=ct_image,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, mask_slice], iou_prediction[:, mask_slice]

    def predict_masks(
        self,
        image_embeddings,
        ct_image,
        text_embedding,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ):
        base = self.base_decoder
        output_tokens = torch.cat(
            [base.iou_token.weight, base.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        if image_embeddings.shape[0] != tokens.shape[0]:
            source = torch.repeat_interleave(
                image_embeddings, tokens.shape[0], dim=0
            )
        else:
            source = image_embeddings
        source = source + dense_prompt_embeddings
        positional_source = torch.repeat_interleave(
            image_pe, tokens.shape[0], dim=0
        )
        batch, channels, depth, height, width = source.shape
        hidden, source = base.transformer(
            source, positional_source, tokens
        )
        iou_token_output = hidden[:, 0]
        mask_token_output = hidden[:, 1 : 1 + base.num_mask_tokens]
        source = source.transpose(1, 2).view(
            batch, channels, depth, height, width
        )
        coarse_feature = base.output_upscaling(source)
        fused_feature = self._fuse(coarse_feature, ct_image)
        hypernetwork_outputs = torch.stack([
            base.output_hypernetworks_mlps[index](
                mask_token_output[:, index]
            )
            for index in range(base.num_mask_tokens)
        ], dim=1)
        batch, channels, depth, height, width = fused_feature.shape
        flattened = fused_feature.view(
            batch, channels, depth * height * width
        )
        masks = (hypernetwork_outputs @ flattened).view(
            batch, -1, depth, height, width
        )
        if text_embedding is not None:
            projected_text = base.txt_align_upscaled_embedding(
                text_embedding
            ).unsqueeze(1)
            similarity = (projected_text @ flattened).view(
                batch, -1, depth, height, width
            )
            masks = masks + similarity.repeat(
                1, masks.shape[1], 1, 1, 1
            )
        iou_prediction = base.iou_prediction_head(iou_token_output)
        return masks, iou_prediction


class CTSkipSegVol(nn.Module):
    """SegVol-compatible wrapper that passes native CT to the skip decoder."""

    def __init__(self, base_segvol: nn.Module, fusion: str):
        super().__init__()
        self.image_encoder = base_segvol.image_encoder
        self.prompt_encoder = base_segvol.prompt_encoder
        self.feat_shape = np.asarray(base_segvol.feat_shape)
        self.mask_decoder = CTSkipMaskDecoder(
            base_segvol.mask_decoder, fusion
        )

    def forward(
        self, image, text_emb=None, text=None,
        boxes=None, points=None,
    ):
        batch = image.shape[0]
        image_shape = tuple(image.shape[-3:])
        image_embedding, _ = self.image_encoder(image)
        expected_tokens = int(np.prod(self.feat_shape))
        if image_embedding.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} image tokens, "
                f"got {image_embedding.shape[1]}"
            )
        image_embedding = image_embedding.transpose(1, 2).view(
            batch,
            -1,
            int(self.feat_shape[0]),
            int(self.feat_shape[1]),
            int(self.feat_shape[2]),
        )
        sparse, dense = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embedding=text_emb,
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
            low_resolution_masks,
            size=image_shape,
            mode="trilinear",
            align_corners=False,
        )


def configure_ct_skip_variant(model, mode: str) -> dict:
    if mode not in {"baseline", "ct_skip_add", "ct_skip_gated"}:
        raise ValueError(mode)
    core = model.get_model()
    before_total = sum(parameter.numel() for parameter in model.parameters())
    base_segvol = core.seg_module
    base_segvol_parameters = sum(
        parameter.numel() for parameter in base_segvol.parameters()
    )
    base_decoder_parameters = sum(
        parameter.numel()
        for parameter in base_segvol.mask_decoder.parameters()
    )
    if mode != "baseline":
        fusion = "addition" if mode == "ct_skip_add" else "gated"
        wrapper = CTSkipSegVol(base_segvol, fusion)
        reference = next(base_segvol.parameters())
        wrapper.to(device=reference.device, dtype=reference.dtype)
        core.seg_module = wrapper
    after_total = sum(parameter.numel() for parameter in model.parameters())
    added = after_total - before_total
    random_weight_parameters = 0
    deterministic_new_parameters = 0
    new_parameter_names = []
    if mode != "baseline":
        for name, parameter in core.seg_module.named_parameters():
            if "ct_skip_stem" in name or ".gate." in name:
                new_parameter_names.append(name)
                if name.endswith("weight") and "norm" not in name:
                    random_weight_parameters += parameter.numel()
                else:
                    deterministic_new_parameters += parameter.numel()
    return {
        "mode": mode,
        "fusion": (
            "none" if mode == "baseline"
            else "addition" if mode == "ct_skip_add"
            else "sigmoid_gated_addition"
        ),
        "preexisting_model_parameters_reused": before_total,
        "pretrained_segvol_parameters_reused": base_segvol_parameters,
        "pretrained_mask_decoder_parameters_reused": base_decoder_parameters,
        "added_parameters": added,
        "random_weight_parameters": random_weight_parameters,
        "deterministically_initialized_bias_or_norm_parameters": (
            deterministic_new_parameters
        ),
        "pretrained_reuse_ratio": before_total / after_total,
        "new_parameter_names": new_parameter_names,
        "pretrained_modules_modified": False,
    }
