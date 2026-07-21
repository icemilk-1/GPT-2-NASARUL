"""Hybrid Prompt RUL Model — Soft Prompt Injection into Frozen GPT-2.

Route B (Hybrid):  PCF Block → Linear(128→768) → [Soft Prompts; Data] → Frozen GPT-2 → Output
Route A (Text):    Sensor Text → Tokenizer → Embedding → [Soft Prompts; Text] → Frozen GPT-2 → Output

Key idea from Bian et al. (IEEE TTE 2025): Soft prompts act as trainable "control knobs"
that dynamically condition the frozen LLM's hidden states for task-specific adaptation.

Only soft prompts, PCF blocks, linear mapping, and output head are trainable.
GPT-2 is fully frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

from src.pcf_block import PCFBlock

logger = logging.getLogger(__name__)


# =============================================================================
#  Hybrid Prompt RUL — Route B (PCF → Soft Prompts → GPT-2)
# =============================================================================

class HybridPromptRUL(nn.Module):
    """GPT4RUL extended with learnable soft prompt vectors (prefix tuning).

    Architecture:
      Sensor Input (B, T, F)
        → Patchify → (B, P, K*F)
        → Input Proj + Pos Encoding → (B, P, 128)
        → PCF Blocks × N → (B, P, 128)
        → Linear Mapping → (B, P, 768)
        → [Soft Prompts; Data Features] → (B, M+P, 768)
        → Frozen GPT-2 (3 layers) → (B, M+P, 768)
        → Slice data portion → (B, P, 768)
        → Flatten & LayerNorm → Linear(1) → RUL

    Only PCF, linear mapping, soft prompts, and output head are trained.
    GPT-2 is frozen.
    """

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
        n_soft_prompts: int = 4,
        gpt2_model_name: str = "openai-community/gpt2",
        gpt2_n_layers: int = 3,
        gpt2_hidden_dim: int = 768,
        freeze_gpt2: bool = True,
        pooling: str = "flatten",
    ):
        super().__init__()
        self.n_features = n_features
        self.window_length = window_length
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.gpt2_hidden_dim = gpt2_hidden_dim
        self.n_soft_prompts = n_soft_prompts
        self.pooling = pooling

        # ---- Compute number of patches ----
        self.num_patches = (window_length - patch_size) // patch_stride + 1
        self.patch_flat_dim = patch_size * n_features
        logger.info(
            "HybridPromptRUL: T=%d K=%d S=%d → P=%d flat_dim=%d soft_prompts=%d",
            window_length, patch_size, patch_stride,
            self.num_patches, self.patch_flat_dim, n_soft_prompts,
        )

        # ===== Stage 1: Patching + Input Embedding =====
        self.input_proj = nn.Linear(self.patch_flat_dim, pcf_hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, pcf_hidden_dim) * 0.02,
        )

        # ===== Stage 2: PCF Blocks =====
        self.pcf_blocks = nn.ModuleList([
            PCFBlock(
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

        # ===== Stage 4: Soft Prompts (Prefix Tuning) =====
        self.soft_prompts = nn.Parameter(
            torch.randn(1, n_soft_prompts, gpt2_hidden_dim) * 0.02,
        )
        logger.info("Soft prompts: %d tokens × %d dim", n_soft_prompts, gpt2_hidden_dim)

        # ===== Stage 5: Frozen GPT-2 =====
        self._build_gpt2(gpt2_model_name, gpt2_n_layers, freeze_gpt2)

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
            self.gpt2: GPT2Model = GPT2Model.from_pretrained(
                str(local_path), local_files_only=True,
            )
        else:
            logger.info("Loading GPT-2 from hub: %s ...", model_name)
            self.gpt2: GPT2Model = GPT2Model.from_pretrained(model_name)

        full_layers = self.gpt2.config.n_layer
        logger.info("GPT-2: %d total layers, using first %d", full_layers, n_layers)

        if n_layers < full_layers:
            self.gpt2.h = self.gpt2.h[-n_layers:]  # last layers: higher-level features
            self.gpt2.config.n_layer = n_layers

        if freeze:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)
            logger.info("GPT-2 parameters frozen.")

        self.gpt2.config.use_cache = False

    # -----------------------------------------------------------------
    #  Patching
    # -----------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, P, K*F) via unfold."""
        B = x.shape[0]
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

        # Stage 1-2: Patchify → PCF (same as GPT4RUL)
        patches = self._patchify(x)
        h = self.input_proj(patches)
        h = h + self.pos_encoding[:, :self.num_patches, :]
        for block in self.pcf_blocks:
            h = block(h)                                         # (B, P, 128)

        # Stage 3: Linear Mapping (128 → 768)
        h = self.linear_mapping(h)                                # (B, P, 768)

        # Stage 4: Prepend soft prompts
        soft = self.soft_prompts.expand(B, -1, -1)                # (B, M, 768)
        h_ext = torch.cat([soft, h], dim=1)                       # (B, M+P, 768)

        # Stage 5: Frozen GPT-2
        gpt2_out = self.gpt2(
            inputs_embeds=h_ext,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = gpt2_out.last_hidden_state                       # (B, M+P, 768)

        # Stage 6: Remove soft prompt outputs, keep data portion
        data_hidden = hidden[:, self.n_soft_prompts:, :]          # (B, P, 768)

        # Stage 7: Pooling + Output
        if self.pooling == "flatten":
            h_pool = data_hidden.reshape(B, -1)                    # (B, P*768)
        elif self.pooling == "mean":
            h_pool = data_hidden.mean(dim=1)                       # (B, 768)
        else:  # "last"
            h_pool = data_hidden[:, -1, :]                         # (B, 768)

        h_pool = self.output_norm(h_pool)
        rul = self.output_head(h_pool).squeeze(-1)                 # (B,)
        return rul

    # -----------------------------------------------------------------
    #  Utilities
    # -----------------------------------------------------------------

    def _log_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        gpt2_total = sum(p.numel() for p in self.gpt2.parameters())
        logger.info(
            "Params: total=%d  trainable=%d  frozen=%d  |  GPT-2=%d",
            total, trainable, frozen, gpt2_total,
        )

    def config_dict(self) -> dict:
        return {
            "model": "HybridPromptRUL",
            "n_features": self.n_features,
            "window_length": self.window_length,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "num_patches": self.num_patches,
            "n_soft_prompts": self.n_soft_prompts,
            "pcf_blocks": len(self.pcf_blocks),
            "gpt2_layers": self.gpt2.config.n_layer,
            "gpt2_hidden": self.gpt2_hidden_dim,
            "pooling": self.pooling,
        }
