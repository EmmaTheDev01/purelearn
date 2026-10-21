"""
Configuration definitions for Purelearn AI Model, Tokenizer, Training, and Inference.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PureLearnConfig:
    """Hyperparameters for the PureLearn Transformer Language Model."""
    vocab_size: int = 4096
    block_size: int = 1024       # Maximum sequence length (context window)
    n_layer: int = 8             # Number of Transformer layers
    n_head: int = 8              # Number of Attention heads
    n_embd: int = 512            # Embedding dimension
    dropout: float = 0.1         # Dropout probability
    bias: bool = False           # Bias in LayerNorms and Linear projections
    use_rotary_emb: bool = True  # Use Rotary Position Embeddings (RoPE)
    norm_eps: float = 1e-5       # RMSNorm/LayerNorm epsilon
    feedforward_dim: Optional[int] = None  # Hidden dimension in MLP (default 4 * n_embd or SwiGLU 8/3 * n_embd)

    def __post_init__(self):
        if self.feedforward_dim is None:
            # SwiGLU standard dimension calculation
            self.feedforward_dim = int(2 * (4 * self.n_embd) / 3)
            # Ensure multiple of 64 for GPU/MPS memory alignment
            self.feedforward_dim = ((self.feedforward_dim + 63) // 64) * 64


@dataclass
class TokenizerConfig:
    """Special tokens and vocabulary settings for PureLearn Tokenizer."""
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    
    # Dialogue role tokens
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    system_token: str = "<|system|>"
    
    # Socratic and Pedagogical tokens
    socratic_question_token: str = "<|socratic_question|>"
    hint_token: str = "<|hint|>"
    scaffold_token: str = "<|scaffold|>"
    concept_check_token: str = "<|concept_check|>"
    
    # Neuro-Inclusive Adaptation tokens
    adhd_chunk_token: str = "<|adhd_chunk|>"
    dyslexia_clear_token: str = "<|dyslexia_clear|>"
    katex_math_token: str = "<|katex_math|>"
    
    # Synthesis & Study Tool tokens
    study_notes_token: str = "<|study_notes|>"
    flashcard_token: str = "<|flashcard|>"

    def get_special_tokens(self) -> List[str]:
        return [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.user_token,
            self.assistant_token,
            self.system_token,
            self.socratic_question_token,
            self.hint_token,
            self.scaffold_token,
            self.concept_check_token,
            self.adhd_chunk_token,
            self.dyslexia_clear_token,
            self.katex_math_token,
            self.study_notes_token,
            self.flashcard_token,
        ]


@dataclass
class TrainingConfig:
    """Hyperparameters for training the PureLearn model."""
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 2000
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 250
    device: Optional[str] = None  # 'cuda', 'mps', or 'cpu' (auto-selected if None)
