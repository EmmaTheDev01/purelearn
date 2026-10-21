"""
PureLearn Custom Subword / BPE Tokenizer with Socratic and Neuro-Inclusive Special Tokens.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from .config import TokenizerConfig


class PureLearnTokenizer:
    """
    Custom subword tokenizer tailored for educational text, KaTeX STEM formulas,
    and neuro-inclusive special markers (ADHD micro-chunks, Dyslexia phonetic anchors, Socratic scaffolds).
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.special_tokens = self.config.get_special_tokens()
        
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Regex pattern for splitting text preserving whitespace, LaTeX math, words, numbers, and punctuation
        self.pattern = re.compile(
            r"""<\|[\w_]+\|>|"""            # Special tokens like <|im_start|>, <|socratic_question|>
            r"""\\[a-zA-Z]+|"""             # LaTeX commands e.g. \frac, \sqrt, \alpha
            r"""\$\$.*?\$\$|\$.*?\$|"""     # Inline or block math formulas
            r"""\d+(?:\.\d+)?|"""           # Numbers (integers & floats)
            r"""[^\r\n\s\w]?[a-zA-Z]+|"""   # Words with optional leading symbol
            r"""\s+|"""                     # Whitespace
            r"""[^\s\w]"""                  # Any other punctuation / symbols
            , re.UNICODE | re.DOTALL
        )
        
        self._init_default_vocabulary()

    def _init_default_vocabulary(self):
        """Initializes the base vocabulary with special tokens, bytes, and core educational subwords."""
        # 1. Register special tokens
        for token in self.special_tokens:
            self._add_token(token)

        # 2. Register all 256 byte characters (for byte-level fallback)
        for i in range(256):
            byte_char = chr(i)
            if byte_char not in self.token_to_id:
                self._add_token(byte_char)

        # 3. Add educational, scientific, mathematical, and neurodivergent frequent terms
        common_terms = [
            # Socratic prompts & framing
            "What", "Why", "How", "Explain", "Consider", "Notice", "Recall", "Observe",
            "What if", "Can you find", "Let's break this down", "Step 1", "Step 2", "Step 3",
            "Good job", "Great insight", "Let's explore", "What do you think happens next?",
            
            # STEM & Math
            "equation", "formula", "derivative", "integral", "function", "variable",
            "theorem", "hypothesis", "experiment", "molecule", "reaction", "energy",
            "velocity", "acceleration", "force", "gravity", "algorithm", "complexity",
            "matrix", "vector", "probability", "triangle", "circle", "angle", "pi",
            "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\partial", "\\infty",
            "\\theta", "\\lambda", "\\Delta", "\\times", "\\cdot", "\\approx", "\\neq",
            
            # Neuro-Inclusive / Study Tools
            "Summary", "Key Takeaway", "Concept Check", "Flashcard", "Question", "Answer",
            "Tip", "Important", "Memory Aid", "Core Definition", "Quick Recap",
            
            # Common English subwords & punctuation
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
        ]

        for term in common_terms:
            if term not in self.token_to_id:
                self._add_token(term)

    def _add_token(self, token: str) -> int:
        """Adds a token to the vocabulary if not already present."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            return token_id
        return self.token_to_id[token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.config.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.config.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.config.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.config.eos_token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes a string into a list of token IDs."""
        if not text:
            return []

        tokens: List[str] = []
        # Find all tokens using the pattern
        matches = self.pattern.findall(text)
        
        for piece in matches:
            if piece in self.token_to_id:
                tokens.append(piece)
            else:
                # Sub-chunk or byte-level fallback
                i = 0
                while i < len(piece):
                    matched = False
                    # Greedily match longest prefix in vocabulary
                    for end in range(len(piece), i, -1):
                        sub = piece[i:end]
                        if sub in self.token_to_id:
                            tokens.append(sub)
                            i = end
                            matched = True
                            break
                    if not matched:
                        # Fallback to byte char
                        char = piece[i]
                        if char in self.token_to_id:
                            tokens.append(char)
                        else:
                            tokens.append(self.config.unk_token)
                        i += 1

        token_ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]

        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes a list of token IDs into a human-readable string."""
        pieces: List[str] = []
        for tid in token_ids:
            if tid not in self.id_to_token:
                continue
            token_str = self.id_to_token[tid]
            if skip_special_tokens and token_str in self.special_tokens:
                continue
            pieces.append(token_str)
        return "".join(pieces)

    def train_from_texts(self, texts: List[str], target_vocab_size: int = 4096):
        """
        Builds and expands vocabulary from a list of educational texts using subword frequency mining.
        """
        from collections import Counter
        word_counts = Counter()

        for text in texts:
            matches = self.pattern.findall(text)
            for word in matches:
                if len(word) > 1 and word not in self.token_to_id:
                    word_counts[word] += 1

        # Add most frequent words/subwords up to target_vocab_size
        most_common = word_counts.most_common(target_vocab_size - len(self.token_to_id))
        for word, _ in most_common:
            self._add_token(word)

    def save(self, filepath: Union[str, Path]):
        """Saves the tokenizer vocabulary and config to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PureLearnTokenizer":
        """Loads a saved tokenizer from a JSON file."""
        tokenizer = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tokenizer.special_tokens = data.get("special_tokens", tokenizer.config.get_special_tokens())
        return tokenizer
