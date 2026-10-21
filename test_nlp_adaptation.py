"""
Unit & Integration Test Suite for PureLearn NLP Human Communication & Affect Adaptation.
Validates intent, affect, hesitation, misconception recognition, empathetic Socratic scaffolding,
and ensures strict 0-emoji guarantee across all communication layers.
"""

import re
import unittest
from purelearn_model.engine import HumanCommunicationAnalyzer, PureLearnEngine
from purelearn_model.tokenizer import PureLearnTokenizer
from purelearn_model.config import PureLearnConfig
from purelearn_model.model import PureLearnLM

EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf\u2300-\u23ff\u2b50\u2b55\u200d\ufe0f]"
)


class TestHumanCommunicationAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = HumanCommunicationAnalyzer()

    def test_confusion_detection(self):
        res = self.analyzer.analyze("I don't understand this at all, I'm completely lost.")
        self.assertTrue(res["is_confused"])
        self.assertIn("ADAPTATION", res["adaptation_guidance"])
        self.assertEqual(res["primary_affect"], "confused")

    def test_frustration_detection(self):
        res = self.analyzer.analyze("This is impossible, my brain hurts and I want to give up.")
        self.assertTrue(res["is_frustrated"])
        self.assertEqual(res["primary_affect"], "frustrated")
        self.assertIn("calming validation", res["adaptation_guidance"])

    def test_tentativeness_detection(self):
        res = self.analyzer.analyze("Maybe the derivative is 2x, but I'm not really sure?")
        self.assertTrue(res["is_tentative"])
        self.assertEqual(res["primary_affect"], "tentative")
        self.assertIn("tentative hypothesis", res["adaptation_guidance"])

    def test_analogy_seeking(self):
        res = self.analyzer.analyze("Can you give me an everyday real-world analogy to picture this?")
        self.assertTrue(res["is_analogy"])
        self.assertIn("visualization", res["adaptation_guidance"])

    def test_misconception_fall_faster(self):
        res = self.analyzer.analyze("Heavier objects always fall faster because they have more mass.")
        self.assertEqual(res["misconception_type"], "heavier_falls_faster")

    def test_misconception_speed_acceleration(self):
        res = self.analyzer.analyze("If a car has high velocity, its speed is acceleration.")
        self.assertEqual(res["misconception_type"], "speed_is_acceleration")

    def test_misconception_product_rule(self):
        res = self.analyzer.analyze("The derivative of u*v is u'*v' right?")
        self.assertEqual(res["misconception_type"], "naive_product_derivative")

    def test_verbosity_classification(self):
        terse = self.analyzer.analyze("2x")
        self.assertEqual(terse["verbosity"], "terse")

        conversational = self.analyzer.analyze("I think Newton's first law applies here because nothing pushes it.")
        self.assertEqual(conversational["verbosity"], "conversational")

        elaborate = self.analyzer.analyze(
            "When we look at the differential equation describing the harmonic oscillator, "
            "the restoring force is proportional to the negative displacement, and the damping term "
            "dissipates mechanical energy into heat as time evolves toward infinity."
        )
        self.assertEqual(elaborate["verbosity"], "elaborate")


class TestNLPEngineAdaptation(unittest.TestCase):
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

    def test_tentative_answer_affirmation(self):
        history = [
            {"role": "user", "content": "How do I differentiate f(x) = x^2 * sin(x)?"},
            {"role": "assistant", "content": "What is the derivative of the first factor, u(x) = x^2?"},
            {"role": "user", "content": "is it maybe 2x?"}
        ]
        res = self.engine.chat(history)
        text = res["response"]
        # Must affirm gently and without emojis
        self.assertIn("spot on", text.lower())
        self.assertFalse(bool(EMOJI_PATTERN.search(text)))
        self.assertIn("sin(x)", text)

    def test_heavier_falls_faster_misconception(self):
        history = [{"role": "user", "content": "Why do heavier objects fall faster than light ones?"}]
        res = self.engine.chat(history)
        text = res["response"]
        self.assertFalse(bool(EMOJI_PATTERN.search(text)))
        self.assertIn("Aristotle", text)
        self.assertIn("mass", text)

    def test_frustration_handling(self):
        history = [{"role": "user", "content": "I hate this, I give up, this is too much"}]
        res = self.engine.chat(history)
        text = res["response"]
        self.assertFalse(bool(EMOJI_PATTERN.search(text)))
        self.assertIn("breath", text.lower())

    def test_idk_gentle_clue(self):
        history = [{"role": "user", "content": "idk"}]
        res = self.engine.chat(history)
        text = res["response"]
        self.assertFalse(bool(EMOJI_PATTERN.search(text)))
        self.assertIn("completely fine", text.lower())

    def test_zero_emojis_in_all_modes(self):
        for adhd, dyslexia in [(True, False), (False, True), (True, True)]:
            res = self.engine.chat(
                [{"role": "user", "content": "Explain binary search"}],
                adhd_mode=adhd,
                dyslexia_mode=dyslexia,
            )
            self.assertFalse(bool(EMOJI_PATTERN.search(res["response"])))


if __name__ == "__main__":
    unittest.main()
