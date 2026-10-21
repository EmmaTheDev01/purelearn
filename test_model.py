"""
Comprehensive Test Suite for PureLearn PyTorch AI Model, Tokenizer, and Engine.
"""

import os
import unittest
import torch

from purelearn_model.config import PureLearnConfig, TokenizerConfig, TrainingConfig
from purelearn_model.dataset import (
    PureLearnDataset,
    collate_purelearn_batch,
    get_default_curriculum_texts,
)
from purelearn_model.engine import PureLearnEngine
from purelearn_model.model import CausalSelfAttention, PureLearnLM, RMSNorm, RotaryEmbedding
from purelearn_model.tokenizer import PureLearnTokenizer


class TestPureLearnTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PureLearnTokenizer()

    def test_special_tokens_registered(self):
        special_tokens = self.tokenizer.config.get_special_tokens()
        for token in special_tokens:
            self.assertIn(token, self.tokenizer.token_to_id)
            self.assertEqual(self.tokenizer.id_to_token[self.tokenizer.token_to_id[token]], token)

    def test_encode_decode_roundtrip(self):
        text = "Hello Purelearn! Let us solve $\\frac{d}{dx}[x^2]$ step by step."
        token_ids = self.tokenizer.encode(text)
        self.assertTrue(len(token_ids) > 0)
        decoded = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded, text)

    def test_special_token_encoding(self):
        text = "<|im_start|><|user|>What is gravity?<|im_end|><|assistant|><|socratic_question|>What causes objects to fall?<|im_end|>"
        token_ids = self.tokenizer.encode(text)
        self.assertIn(self.tokenizer.bos_id, token_ids)
        self.assertIn(self.tokenizer.eos_id, token_ids)
        self.assertIn(self.tokenizer.token_to_id["<|socratic_question|>"], token_ids)


class TestPureLearnModel(unittest.TestCase):
    def setUp(self):
        self.config = PureLearnConfig(
            vocab_size=256,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
        )
        self.model = PureLearnLM(self.config)

    def test_rmsnorm(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_rotary_embedding(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        q = torch.randn(2, 2, 10, 32)
        k = torch.randn(2, 2, 10, 32)
        q_rot, k_rot = rope(q, k, seq_len=10)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_forward_pass_and_loss(self):
        batch_size = 2
        seq_len = 16
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        logits, loss, _ = self.model(x, targets=targets)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

        # Test backward pass
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)

    def test_generation(self):
        prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
        out = self.model.generate(prompt, max_new_tokens=5, temperature=0.8, top_p=0.9)
        self.assertEqual(out.shape, (1, 8))

    def test_streaming_generation(self):
        prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
        stream_tokens = list(self.model.generate_stream(prompt, max_new_tokens=4, temperature=0.7))
        self.assertEqual(len(stream_tokens), 4)


class TestPureLearnDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PureLearnTokenizer()
        self.texts = get_default_curriculum_texts(self.tokenizer)

    def test_dataset_and_collate(self):
        dataset = PureLearnDataset(self.texts, self.tokenizer, max_seq_len=128)
        self.assertTrue(len(dataset) > 0)
        x, y = dataset[0]
        self.assertEqual(len(x), len(y))

        batch = [dataset[0], dataset[1]]
        padded_x, padded_y = collate_purelearn_batch(batch, pad_id=self.tokenizer.pad_id)
        self.assertEqual(padded_x.shape[0], 2)
        self.assertEqual(padded_y.shape[0], 2)
        self.assertEqual(padded_x.shape[1], padded_y.shape[1])


class TestPureLearnEngine(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PureLearnTokenizer()
        self.config = PureLearnConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=128,
            n_layer=2,
            n_head=2,
            n_embd=64,
        )
        self.model = PureLearnLM(self.config)
        self.engine = PureLearnEngine(model=self.model, tokenizer=self.tokenizer, device="cpu")

    def test_prompt_construction(self):
        messages = [{"role": "user", "content": "Help me understand calculus."}]
        prompt = self.engine.build_socratic_prompt(
            messages, socratic_level=2, adhd_mode=True, dyslexia_mode=False
        )
        self.assertIn("<|system|>", prompt)
        self.assertIn("ADHD Mode Active", prompt)
        self.assertIn("<|user|>Help me understand calculus.<|im_end|>", prompt)

    def test_bionic_reading(self):
        sample = "Understand the fundamental theorem of calculus"
        bionic = self.engine.apply_bionic_reading(sample)
        self.assertIn("**", bionic)

    def test_study_notes_and_flashcards(self):
        notes = self.engine.generate_study_notes("Newtonian Mechanics and Momentum")
        self.assertIn("notes_markdown", notes)
        self.assertTrue(len(notes["notes_markdown"]) > 0)

        flashcards = self.engine.generate_flashcards("Newtonian Mechanics")
        self.assertIsInstance(flashcards, list)
        self.assertTrue(len(flashcards) > 0)
        self.assertIn("question", flashcards[0])
        self.assertIn("answer", flashcards[0])


if __name__ == "__main__":
    unittest.main()
