"""PCF (Patch-Channel Fusion) Block — shared module.

MLP-Mixer inspired lightweight adapter for mixing information across time
(patches) and features (channels). Used by GPT4RUL and HybridPromptRUL.

Reference: Tan et al., QR2MSE 2025, Section 2.3.3.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PCFBlock(nn.Module):
    """Patch-Channel Fusion block (Section 2.3.3), inspired by MLP-Mixer [14].

    Architecture (MLP-Mixer style, Pre-LN, sequential):
      1) LayerNorm → Patch-Mixing MLP (across patches) → Residual Add
      2) LayerNorm → Channel-Mixing MLP (across channels) → Residual Add

    Path1 (Patch-Mixing): transpose(P,D) → MLP across P → transpose back
    Path2 (Channel-Mixing): MLP across D directly

    Mathematical formulation:
      Given input x ∈ R^{B×P×D} (B=batch, P=patches, D=channels):

      Step 1 — Patch-Mixing:
        x_norm = LayerNorm(x)                    # (B, P, D)
        x_trans = x_norm.transpose(1, 2)         # (B, D, P)
        x_mixed = MLP_patch(x_trans)             # (B, D, P): mix across patch dim
        x_mixed = x_mixed.transpose(1, 2)        # (B, P, D)
        x = x + x_mixed                          # Residual connection

      Step 2 — Channel-Mixing:
        x_norm = LayerNorm(x)                    # (B, P, D)
        x_mixed = MLP_channel(x_norm)            # (B, P, D): mix across channel dim
        x = x + x_mixed                          # Residual connection

      MLP_patch:  P → (P·mf) → GELU → Dropout → P → Dropout
      MLP_channel: D → (D·mf) → GELU → Dropout → D → Dropout

      where mf = mixing_factor controls the hidden expansion ratio.
    """

    def __init__(
        self,
        dim: int = 128,
        num_patches: int = 6,
        mixing_factor: float = 2.0,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Independent LayerNorms for Pre-LN (standard MLP-Mixer [14])
        self.patch_norm = nn.LayerNorm(dim)
        self.channel_norm = nn.LayerNorm(dim)

        # ---- Patch-Mixing path ----
        # Operates on (B, D, P): mix across P patches
        patch_hidden = int(num_patches * mixing_factor)
        self.patch_mlp = nn.Sequential(
            nn.Linear(num_patches, patch_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_hidden, num_patches),
            nn.Dropout(dropout),
        )

        # ---- Channel-Mixing path ----
        # Operates on (B, P, D): mix across D channels
        channel_hidden = int(dim * mixing_factor)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P, D) → (B, P, D)"""
        # ---- Step 1: Patch-Mixing (Pre-LN) ----
        n = self.patch_norm(x)                             # (B, P, D) Pre-LN
        p = n.transpose(1, 2)                             # (B, D, P)
        p = self.patch_mlp(p)                              # (B, D, P)
        p = p.transpose(1, 2)                             # (B, P, D)
        x = x + p                                         # Residual Add

        # ---- Step 2: Channel-Mixing (Pre-LN) ----
        n = self.channel_norm(x)                           # (B, P, D) Pre-LN
        c = self.channel_mlp(n)                            # (B, P, D)
        x = x + c                                         # Residual Add

        return x


class PCFBlockParallel(nn.Module):
    """Patch-Channel Fusion: dual-path parallel MLP + additive fusion (alternative).

    Unlike the sequential variant, patch-mixing and channel-mixing operate
    in parallel on the original input, with independent LayerNorms and
    additive fusion at the output.

    Mathematical formulation:
      Given input x ∈ R^{B×P×D}:

      Patch path:
        p = x.transpose(1, 2)                 # (B, D, P)
        p = MLP_patch(p).transpose(1, 2)       # (B, P, D)
        p = LayerNorm(p + x)                   # Residual + Norm

      Channel path:
        c = MLP_channel(x)                     # (B, P, D)
        c = LayerNorm(c + x)                   # Residual + Norm

      Output: p + c                            # Additive fusion
    """

    def __init__(
        self,
        dim: int = 128,
        num_patches: int = 6,
        mixing_factor: float = 2.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        patch_hidden = int(num_patches * mixing_factor)
        self.patch_mlp = nn.Sequential(
            nn.Linear(num_patches, patch_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_hidden, num_patches),
            nn.Dropout(dropout),
        )
        self.patch_norm = nn.LayerNorm(dim)
        channel_hidden = int(dim * mixing_factor)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, dim),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x.transpose(1, 2)
        p = self.patch_mlp(p).transpose(1, 2)
        p = self.patch_norm(p + x)
        c = self.channel_mlp(x)
        c = self.channel_norm(c + x)
        return p + c
