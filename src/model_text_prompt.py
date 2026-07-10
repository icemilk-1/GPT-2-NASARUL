"""Text Prompt RUL Model — Route A (Pure Prompt).

Converts sensor windows to natural language text, feeds through GPT-2 tokenizer,
with learnable soft prompt prefix tokens (prefix tuning).

Architecture:
  Sensor window (B, T, F)
    → text template (window_to_text)
    → GPT-2 tokenizer → input_ids (B, L)
    → GPT-2 embedding → (B, L, 768)
    → [Soft Prompts; Text Embeddings] → (B, M+L, 768)
    → Frozen GPT-2 → (B, M+L, 768)
    → Slice text portion → take "answer" position
    → Linear → RUL

Only soft prompts and output head are trainable. GPT-2 is fully frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

logger = logging.getLogger(__name__)


class TextPromptRUL(nn.Module):
    """Route A: pure text prompt + soft prompts + frozen GPT-2 for RUL."""

    def __init__(
        self,
        gpt2_model_name: str = "openai-community/gpt2",
        gpt2_n_layers: int = 3,
        gpt2_hidden_dim: int = 768,
        n_soft_prompts: int = 4,
        freeze_gpt2: bool = True,
        max_length: int = 512,
    ):
        super().__init__()
        self.gpt2_hidden_dim = gpt2_hidden_dim
        self.n_soft_prompts = n_soft_prompts
        self.max_length = max_length

        # ---- GPT-2 Tokenizer ----
        local_path = Path(__file__).resolve().parent.parent / ".hf_cache" / "gpt2"
        if local_path.exists():
            self.tokenizer = GPT2Tokenizer.from_pretrained(str(local_path), local_files_only=True)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Frozen GPT-2 ----
        self._build_gpt2(gpt2_model_name, gpt2_n_layers, freeze_gpt2)

        # ---- Soft Prompts (prefix tuning) ----
        self.soft_prompts = nn.Parameter(
            torch.randn(1, n_soft_prompts, gpt2_hidden_dim) * 0.02,
        )
        logger.info("TextPromptRUL: soft_prompts=%d tokens × %d dim", n_soft_prompts, gpt2_hidden_dim)

        # ---- Output head ----
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
            self.gpt2.h = self.gpt2.h[:n_layers]
            self.gpt2.config.n_layer = n_layers

        if freeze:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)
            logger.info("GPT-2 parameters frozen.")

        self.gpt2.config.use_cache = False

    # -----------------------------------------------------------------
    #  Forward
    # -----------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """input_ids: (B, L) → rul: (B,)"""
        B, L = input_ids.shape

        # Get GPT-2 embeddings for text tokens
        text_embeds = self.gpt2.wte(input_ids)                      # (B, L, 768)

        # Prepend soft prompts
        soft = self.soft_prompts.expand(B, -1, -1)                  # (B, M, 768)
        h_ext = torch.cat([soft, text_embeds], dim=1)               # (B, M+L, 768)

        # Build attention mask (1 for all tokens, including soft prompts)
        if attention_mask is not None:
            soft_mask = torch.ones(B, self.n_soft_prompts, device=input_ids.device)
            attn_mask = torch.cat([soft_mask, attention_mask], dim=1)
        else:
            attn_mask = torch.ones(B, self.n_soft_prompts + L, device=input_ids.device)

        # Frozen GPT-2
        gpt2_out = self.gpt2(
            inputs_embeds=h_ext,
            attention_mask=attn_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = gpt2_out.last_hidden_state                        # (B, M+L, 768)

        # Take the last token's hidden state as the "answer" representation
        last_hidden = hidden[:, -1, :]                              # (B, 768)

        # Output head
        h_norm = self.output_norm(last_hidden)
        rul = self.output_head(h_norm).squeeze(-1)                  # (B,)
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
            "model": "TextPromptRUL",
            "gpt2_layers": self.gpt2.config.n_layer,
            "gpt2_hidden": self.gpt2_hidden_dim,
            "n_soft_prompts": self.n_soft_prompts,
            "max_length": self.max_length,
        }
