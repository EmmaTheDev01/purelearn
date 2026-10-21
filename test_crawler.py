"""
Unit & Integration Test Suite for PureLearn Internet Crawler and Data Pipeline.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from purelearn_model.crawler import (
    CrawlerConfig,
    InternetDataPipeline,
    SocraticDataSynthesizer,
    WebTextExtractor,
)
from purelearn_model.dataset import (
    PackedPureLearnDataset,
    PureLearnDataset,
    create_train_val_split,
)
from purelearn_model.tokenizer import PureLearnTokenizer


class TestWebTextExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = WebTextExtractor(extract_math=True)

    def test_html_cleaning_and_formula_extraction(self):
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Calculus: The Derivative - Wikipedia</title></head>
        <body>
            <nav><a href="/home">Home</a></nav>
            <header>Site Header</header>
            <main>
                <h1>The Derivative</h1>
                <p>In mathematics, the derivative of a function measures its sensitivity to change.</p>
                <p>The derivative formula is given by: <span class="mwe-math-element"><annotation encoding="application/x-tex">f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}</annotation></span>.</p>
                <ul>
                    <li>First order derivative represents rate of change.</li>
                    <li>Second order derivative represents curvature.</li>
                </ul>
            </main>
            <footer>Copyright 2026</footer>
            <script>alert("ad");</script>
        </body>
        </html>
        """
        extracted = self.extractor.clean_html(sample_html)
        self.assertEqual(extracted["title"], "Calculus: The Derivative")
        self.assertIn("sensitivity to change", extracted["text"])
        self.assertNotIn("Copyright 2026", extracted["text"])
        self.assertNotIn("Site Header", extracted["text"])
        self.assertNotIn("alert", extracted["text"])
        self.assertTrue(len(extracted["formulas"]) > 0)
        self.assertIn("\\lim_{h \\to 0}", extracted["formulas"][0])

    def test_deduplication(self):
        text = "Calculus is the mathematical study of continuous change, in the same way that geometry is the study of shape."
        self.assertFalse(self.extractor.is_duplicate(text))
        self.assertTrue(self.extractor.is_duplicate(text))


class TestSocraticSynthesizer(unittest.TestCase):
    def setUp(self):
        self.synthesizer = SocraticDataSynthesizer()

    def test_synthesize_socratic_dialogue(self):
        article = {
            "title": "Newton's Second Law",
            "text": "Newton's second law states that the acceleration of an object depends directly upon the net force. As mass increases, acceleration decreases for a given force.",
            "formulas": [r"F = m \cdot a"],
            "url": "https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion",
        }
        res = self.synthesizer.synthesize_socratic_dialogue(article, variant_id=1)
        self.assertEqual(res["concept"], "Newton's Second Law")
        self.assertIn("dialogue", res)
        dialogue = res["dialogue"]
        self.assertEqual(len(dialogue), 7)


        # Check special tokens in assistant responses
        assistant_turn = dialogue[2]["content"]
        self.assertIn("<|adhd_chunk|>", assistant_turn)
        self.assertIn("<|socratic_question|>", assistant_turn)
        self.assertIn("<|katex_math|>", assistant_turn)

        final_turn = dialogue[6]["content"]
        self.assertIn("<|study_notes|>", final_turn)

    def test_synthesize_pretraining_document(self):
        article = {
            "title": "Binary Search",
            "text": "Binary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing in half the portion of the list.",
            "url": "https://en.wikipedia.org/wiki/Binary_search_algorithm",
        }
        doc = self.synthesizer.synthesize_pretraining_document(article)
        self.assertEqual(doc["title"], "Binary Search")
        self.assertIn("<|system|>", doc["text"])
        self.assertIn("<|user|>", doc["text"])
        self.assertIn("<|assistant|>", doc["text"])
        self.assertIn("<|im_end|>", doc["text"])


class TestPackedDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PureLearnTokenizer()
        self.texts = [
            "<|system|>You are Purelearn AI.<|im_end|>\n<|user|>What is force?<|im_end|>\n<|assistant|>Force is mass times acceleration.<|im_end|>",
            "<|system|>You are Purelearn AI.<|im_end|>\n<|user|>What is momentum?<|im_end|>\n<|assistant|>Momentum is mass times velocity.<|im_end|>",
        ]

    def test_packed_dataset_chunks(self):
        block_size = 16
        dataset = PackedPureLearnDataset(self.texts, self.tokenizer, block_size=block_size)
        self.assertTrue(len(dataset) > 0)
        x, y = dataset[0]
        self.assertEqual(len(x), block_size)
        self.assertEqual(len(y), block_size)
        # Shift check: y[i] should be x[i+1]
        self.assertTrue((x[1:] == y[:-1]).all())

    def test_train_val_split(self):
        texts = [f"Text sequence {i}" for i in range(20)]
        train_texts, val_texts = create_train_val_split(texts, val_ratio=0.2, seed=42)
        self.assertEqual(len(train_texts), 16)
        self.assertEqual(len(val_texts), 4)


class TestPipelineExport(unittest.TestCase):
    def test_export_and_load(self):
        pipeline = InternetDataPipeline()
        articles = [
            {
                "title": "Quantum Entanglement",
                "text": "Quantum entanglement is the phenomenon where a pair or group of particles interact such that each particle cannot be described independently.",
                "formulas": [r"|\psi\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}"],
                "url": "https://en.wikipedia.org/wiki/Quantum_entanglement",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_crawled.jsonl")
            count = pipeline.process_and_export(articles, output_file=out_file, generate_socratic=True)
            self.assertEqual(count, 2)  # 1 dialogue + 1 doc

            with open(out_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertIn("dialogue", lines[0])
            self.assertIn("text", lines[1])


if __name__ == "__main__":
    unittest.main()
