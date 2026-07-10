"""GPT4RUL Model — strict paper reproduction (Tan et al., QR2MSE 2025).

Architecture (Section 2.3, Fig.1):
  Input (B, T, F)
    → Patching: unfold → (B, P, F*K)  [K=8, stride per dataset]
    → Linear Proj: (F*K → 128) + Learnable Position Encoding
    → PCF Block × N (MLP-Mixer style, Pre-LN sequential):
        ├─ LayerNorm → Patch-Mixing MLP → Residual Add
        ├─ LayerNorm → Channel-Mixing MLP → Residual Add
    → Linear Mapping: 128 → 768
    → Frozen GPT-2 (3 layers only)
    → Add & LayerNorm (residual around GPT-2)
    → Flatten & LayerNorm
    → Linear → RUL

Only PCF blocks, embeddings, linear mapping, and output head are trainable.
GPT-2 is fully frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model

from src.pcf_block import PCFBlock, PCFBlockParallel

logger = logging.getLogger(__name__)


# =============================================================================
#  GPT4RUL Main Model
# =============================================================================

class GPT4RUL(nn.Module):
    """Patching + PCF + Frozen GPT-2 for RUL prediction."""

    def __init__(
        self,
        n_features: int,
        window_length: int,
        patch_size: int = 8,
        patch_stride: int = 4,
        pcf_hidden_dim: int = 128,
        n_pcf_blocks: int = 2,
        pcf_mixing_factor: float = 2.0,
        pcf_dropout: float = 0.2,
        pcf_style: str = "sequential",
        gpt2_model_name: str = "openai-community/gpt2",
        gpt2_n_layers: int = 3,
        gpt2_hidden_dim: int = 768,
        freeze_gpt2: bool = True,
        pooling: str = "mean",  # "last", "mean", or "flatten"
        use_gpt2_residual: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.window_length = window_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.gpt2_hidden_dim = gpt2_hidden_dim
        self.pooling = pooling
        self.use_gpt2_residual = use_gpt2_residual
        self.pcf_style = pcf_style

        # ---- Compute number of patches ----
        self.num_patches = (window_length - patch_size) // patch_stride + 1
        self.patch_flat_dim = patch_size * n_features
        logger.info(
            "Patching: T=%d, K=%d, S=%d → P=%d, flat_dim=%d",
            window_length, patch_size, patch_stride,
            self.num_patches, self.patch_flat_dim,
        )

        # ===== Stage 1: Patching + Input Embedding =====
        self.input_proj = nn.Linear(self.patch_flat_dim, pcf_hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, pcf_hidden_dim) * 0.02,
        )

        # ===== Stage 2: PCF Blocks =====
        pcf_cls = PCFBlockParallel if pcf_style == "parallel" else PCFBlock
        self.pcf_blocks = nn.ModuleList([
            pcf_cls(
                dim=pcf_hidden_dim,
                num_patches=self.num_patches,
                mixing_factor=pcf_mixing_factor,
                dropout=pcf_dropout,
            )
            for _ in range(n_pcf_blocks)
        ])

        # ===== Stage 3: Linear Mapping (128 → 768) =====
        self.linear_mapping = nn.Sequential(
            nn.LayerNorm(pcf_hidden_dim),
            nn.Linear(pcf_hidden_dim, gpt2_hidden_dim),
            nn.Dropout(pcf_dropout),
        )

        # ===== Stage 4: Frozen GPT-2 (3 layers) =====
        self._build_gpt2(gpt2_model_name, gpt2_n_layers, freeze_gpt2)

        # ===== Stage 5: Optional residual around GPT-2 =====
        if use_gpt2_residual:
            self.gpt2_residual_norm = nn.LayerNorm(gpt2_hidden_dim)

        # ===== Stage 6: Output head =====
        if pooling == "flatten":
            self.output_norm = nn.LayerNorm(gpt2_hidden_dim * self.num_patches)
            self.output_head = nn.Linear(gpt2_hidden_dim * self.num_patches, 1)
        elif pooling == "mean":
            self.output_norm = nn.LayerNorm(gpt2_hidden_dim)
            self.output_head = nn.Linear(gpt2_hidden_dim, 1)
        else:  # "last"
            self.output_norm = nn.LayerNorm(gpt2_hidden_dim)
            self.output_head = nn.Linear(gpt2_hidden_dim, 1)

        self._log_params()

    # -----------------------------------------------------------------
    #  GPT-2 loading
    # -----------------------------------------------------------------

    def _build_gpt2(self, model_name: str, n_layers: int, freeze: bool) -> None:
        local_path = Path(__file__).resolve().parent.parent / ".hf_cache" / "gpt2"
        if local_path.exists():
            logger.info("Loading GPT-2 from local cache: %s", local_path)
            self.gpt2: GPT2Model = GPT2Model.from_pretrained(str(local_path), local_files_only=True)
        else:
            logger.info("Loading GPT-2 from hub: %s ...", model_name)
            self.gpt2: GPT2Model = GPT2Model.from_pretrained(model_name)

        full_layers = self.gpt2.config.n_layer
        logger.info("GPT-2 loaded: %d total layers, using first %d", full_layers, n_layers)

        if n_layers < full_layers:
            self.gpt2.h = self.gpt2.h[:n_layers]
            self.gpt2.config.n_layer = n_layers

        if freeze:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)
            logger.info("GPT-2 parameters frozen.")

        # Disable cache for gradient checkpointing compatibility
        self.gpt2.config.use_cache = False

    # -----------------------------------------------------------------
    #  Patching
    # -----------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, P, K*F) via unfold."""
        B = x.shape[0]
        # x: (B, T, F) → (B, F, T) → unfold → (B, F, P, K) → (B, P, F, K) → (B, P, K*F)
        x_t = x.transpose(1, 2)
        patches = x_t.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(B, self.num_patches, self.patch_flat_dim)
        return patches.contiguous()

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → rul: (B,)"""
        B = x.shape[0]

        # Stage 1: Patchify + Project + Position Encode
        patches = self._patchify(x)                              # (B, P, K*F)
        h = self.input_proj(patches)                             # (B, P, 128)
        h = h + self.pos_encoding[:, :self.num_patches, :]       # add PE

        # Stage 2: PCF Blocks
        for block in self.pcf_blocks:
            h = block(h)                                         # (B, P, 128)

        # Stage 3: Linear Mapping (128 → 768)
        h = self.linear_mapping(h)                               # (B, P, 768)

        # Stage 4: Frozen GPT-2 (feed via inputs_embeds)
        gpt2_out = self.gpt2(
            inputs_embeds=h,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = gpt2_out.last_hidden_state                      # (B, P, 768)

        # Stage 5: Optional residual around GPT-2
        if self.use_gpt2_residual:
            h_res = h + hidden                                    # (B, P, 768) Add
            h_res = self.gpt2_residual_norm(h_res)                # LayerNorm
        else:
            h_res = hidden

        # Stage 6: Pooling + LayerNorm + Output Linear
        if self.pooling == "flatten":
            h_pool = h_res.reshape(B, -1)                         # (B, P*768)
        elif self.pooling == "mean":
            h_pool = h_res.mean(dim=1)                            # (B, 768)
        else:  # "last"
            h_pool = h_res[:, -1, :]                              # (B, 768)

        h_pool = self.output_norm(h_pool)                         # LayerNorm
        rul = self.output_head(h_pool).squeeze(-1)                # (B,)

        return rul

    # -----------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------

    def _log_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        gpt2_total = sum(p.numel() for p in self.gpt2.parameters())
        gpt2_trainable = sum(
            p.numel() for p in self.gpt2.parameters() if p.requires_grad
        )
        logger.info(
            "Params: total=%d  trainable=%d  frozen=%d  |  GPT-2: total=%d  trainable=%d",
            total, trainable, frozen, gpt2_total, gpt2_trainable,
        )

    def config_dict(self) -> dict:
        return {
            "model": "GPT4RUL",
            "n_features": self.n_features,
            "window_length": self.window_length,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "num_patches": self.num_patches,
            "pcf_blocks": len(self.pcf_blocks),
            "pcf_style": self.pcf_style,
            "gpt2_layers": self.gpt2.config.n_layer,
            "gpt2_hidden": self.gpt2_hidden_dim,
        }
