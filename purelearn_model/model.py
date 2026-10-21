"""
PureLearn Transformer Causal Language Model built from scratch in PyTorch.
Includes:
- RMSNorm
- Rotary Position Embeddings (RoPE)
- Multi-Head Causal Self-Attention with KV-Caching
- SwiGLU Gated Feed-Forward Networks
- Autoregressive Generation with Temperature, Top-P, Top-K, and Streaming
"""

import math
from typing import Generator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PureLearnConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) for enhanced training stability."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) to encode relative spatial & token distances."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Calculate inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Repeat to match dim
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        total_len = offset + seq_len
        if total_len > self.cos_cached.shape[0]:
            self._build_cache(total_len)

        cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
        sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(1)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and KV-Cache support."""

    def __init__(self, config: PureLearnConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.use_rotary = config.use_rotary_emb

        # Q, K, V Projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size)
        else:
            self.rotary_emb = None

        # Causal mask fallback for non-SDPA paths
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()

        # Project and reshape: (B, T, C) -> (B, n_head, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings
        if self.use_rotary and self.rotary_emb is not None:
            if kv_cache is not None:
                past_k, _ = kv_cache
                past_len = past_k.size(2)
                q, k = self.rotary_emb(q, k, seq_len=T, offset=past_len)
            else:
                q, k = self.rotary_emb(q, k, seq_len=T, offset=0)

        # Handle KV Cache for fast autoregressive generation
        new_kv_cache = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if use_cache:
            new_kv_cache = (k, v)

        total_k_len = k.size(2)

        # Accelerated Scaled Dot-Product Attention (SDPA)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        if kv_cache is None and T == total_k_len:
            # Full sequence during training or prefill: native FlashAttention / causal SDPA
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        elif kv_cache is not None and T == 1:
            # Autoregressive decoding step: query attends to all past keys
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        else:
            # Masked fallback
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, total_k_len)
            if T > 1 or kv_cache is None:
                mask = self.causal_mask[:, :, :T, :total_k_len]
                attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return out, new_kv_cache



class SwiGLUFeedForward(nn.Module):
    """SwiGLU Gated Feed-Forward Network."""

    def __init__(self, config: PureLearnConfig):
        super().__init__()
        hidden_dim = config.feedforward_dim or int(4 * config.n_embd)
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(W_gate * x) * (W_up * x)) * W_down
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """Single Transformer Block with Pre-LayerNorm/RMSNorm and residual connections."""

    def __init__(self, config: PureLearnConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention + Residual
        normed_x = self.ln_1(x)
        attn_out, new_kv_cache = self.attn(normed_x, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out

        # Pre-norm MLP + Residual
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache


class PureLearnLM(nn.Module):
    """
    PureLearn Causal Transformer Language Model.
    Designed specifically for Socratic educational dialogue, STEM proofs, and neuro-inclusive formatting.
    """

    def __init__(self, config: PureLearnConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final normalization
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)

        # Language Model output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (share weights between token embeddings and output projection)
        self.tok_embeddings.weight = self.lm_head.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns the total number of trainable parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.
        Args:
            idx: (batch_size, seq_len) token IDs
            targets: Optional (batch_size, seq_len) target token IDs for computing cross entropy loss
            kv_caches: Optional list of cached (k, v) tensors for each layer
            use_cache: Whether to return updated KV caches for fast sampling
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds maximum block size {self.config.block_size}"

        # Token embedding
        x = self.tok_embeddings(idx)
        x = self.drop(x)

        new_kv_caches = [] if use_cache else None

        # Pass through Transformer blocks
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache and updated_cache is not None:
                new_kv_caches.append(updated_cache)

        # Final LayerNorm
        x = self.ln_f(x)

        loss = None
        if targets is not None:
            # If targets are provided, compute cross-entropy loss over all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only compute logits for the last token position to save compute
            logits = self.lm_head(x[:, [-1], :])

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.15,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV-caching."""
        self.eval()
        kv_caches = None

        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
                    logits, _, kv_caches = self(idx_cond, use_cache=True)
                else:
                    logits, _, kv_caches = self(idx[:, [-1]], kv_caches=kv_caches, use_cache=True)
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
                logits, _, _ = self(idx_cond)

            logits = logits[:, -1, :].clone()  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Apply temperature
            if temperature > 0.0:
                logits = logits / max(1e-5, temperature)

                # Top-K Filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-P (Nucleus) Filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (argmax)
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and idx_next.item() == eos_id:
                break

            # Reset cache if context limit is reached
            if use_cache and kv_caches is not None and kv_caches[0][0].size(2) >= self.config.block_size:
                kv_caches = None

        return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.15,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> Generator[int, None, None]:
        """Streaming generator yielding generated token IDs one by one with KV caching."""
        self.eval()
        kv_caches = None

        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
                    logits, _, kv_caches = self(idx_cond, use_cache=True)
                else:
                    logits, _, kv_caches = self(idx[:, [-1]], kv_caches=kv_caches, use_cache=True)
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
                logits, _, _ = self(idx_cond)

            logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            if temperature > 0.0:
                logits = logits / max(1e-5, temperature)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = idx_next.item()
            idx = torch.cat((idx, idx_next), dim=1)

            yield token_id

            if eos_id is not None and token_id == eos_id:
                break

            if use_cache and kv_caches is not None and kv_caches[0][0].size(2) >= self.config.block_size:
                kv_caches = None

