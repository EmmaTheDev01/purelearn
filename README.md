# Purelearn AI Model Engine (PyTorch)

A custom **Transformer Neural Language Model and AI Engine** engineered from scratch in **PyTorch** for [Purelearn.ai](https://purelearn.vigilance.rw) — *Your Neuro-Inclusive Socratic AI Tutor*.

---

## Features

- **Built from Scratch in PyTorch**:
 - Accelerated Multi-Head Causal Self-Attention with PyTorch SDPA (Scaled Dot-Product Attention) and $O(N)$ KV-caching.
 - Rotary Position Embeddings (RoPE) with offset slicing for long-context STEM and mathematical reasoning.
 - Root Mean Square Layer Normalization (RMSNorm) and SwiGLU Gated Feed-Forward blocks.
 - Full autoregressive sampling pipeline (Temperature, Top-K, Top-P Nucleus Filtering, and Repetition Penalty).
- **Internet Data Crawler & Socratic Ingestion Pipeline**:
 - Live internet harvesting from Wikipedia, arXiv scientific papers, and custom educational URLs.
 - Automated HTML & LaTeX math equation extraction ($...$ and $$...$$).
 - Socratic dialogue synthesizer converting raw web articles into neuro-inclusive multi-turn Q&A.
 - High-efficiency packed dataset generation with zero padding waste.
- **Enhanced Training Pipeline**:
 - Continuous validation loss & Perplexity ($e^{\text{loss}}$) tracking.
 - Gradient Accumulation & Automatic Mixed Precision (AMP).
 - Best model checkpointing (`purelearn_best.pt`) and graceful shutdown on interrupt.
- **Neuro-Inclusive Tokenizer & Control Markers**:
 - Socratic pedagogical tokens: `<|socratic_question|>`, `<|hint|>`, `<|scaffold|>`, `<|concept_check|>`
 - Neuro-inclusive formatting tokens: `<|adhd_chunk|>`, `<|dyslexia_clear|>`, `<|katex_math|>`
 - Synthesis tokens: `<|study_notes|>`, `<|flashcard|>`
- **Socratic Tutoring Engine**:
 - 3-level configurable Socratic depth (Gentle Hints &rarr; Guided Discovery &rarr; Strict Inquiry).
 - ADHD cognitive load optimization (micro-chunking, bionic reading fixations, bulleted breakdowns).
 - Dyslexia readability adaptations (high contrast, clear syntax, phonetic disambiguation).
 - KaTeX mathematical formula generation ($...$ and $$...$$).
- **Automated Study Tools**:
 - Instant Markdown study notes synthesizer.
 - Active recall flashcard extractor.
- **FastAPI Production Wrapper**:
 - High-performance REST and Server-Sent Events (SSE) streaming endpoints.
 - Hardware auto-detection: Apple Silicon GPU (`mps`), NVIDIA GPU (`cuda`), or CPU.

---

## Repository Structure

```
purelearn/
├── purelearn_model/
│ ├── __init__.py # Package initialization
│ ├── config.py # Model, tokenizer, and training dataclasses
│ ├── model.py # PyTorch Transformer LM (SDPA, RoPE, RMSNorm, SwiGLU, KV-Cache)
│ ├── tokenizer.py # Subword/BPE tokenizer with special tokens
│ ├── dataset.py # Socratic curriculum generator, Packed dataset & DataLoader
│ ├── crawler.py # Internet web crawler, Wikipedia/arXiv harvester & synthesizer
│ └── engine.py # High-level Socratic & neuro-inclusive inference engine
├── checkpoints/ # Model weights and tokenizer vocabulary
│ ├── tokenizer.json
│ ├── purelearn_latest.pt
│ └── purelearn_best.pt
├── crawl.py # Standalone Internet Web Crawler CLI
├── train.py # Pre-training, fine-tuning & internet ingestion CLI
├── server.py # FastAPI server and streaming API wrapper
├── test_model.py # Transformer and engine unit test suite
├── test_crawler.py # Crawler and packed dataset test suite
├── demo_client.html # Interactive web test interface
├── requirements.txt # Python dependencies
└── README.md # Documentation
```

---

## Quickstart

### 1. Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python3 -m unittest discover -s . -p "test_*.py"
```

### 3. Crawl Internet Data for Training

Harvest Wikipedia articles, arXiv papers, or arbitrary educational URLs, and convert them into Socratic training data:

```bash
# Option A: Crawl specific STEM & Humanities topics
python crawl.py \
 --topics "Calculus,Quantum Mechanics,Algorithms,Neurodiversity" \
 --max_per_topic 3 \
 --output data/crawled_dataset.jsonl

# Option B: Crawl the standard curated STEM curriculum pack
python crawl.py --stem_pack --output data/crawled_stem.jsonl

# Option C: Crawl educational websites or documentation
python crawl.py \
 --urls "https://en.wikibooks.org/wiki/Calculus" \
 --max_pages_per_url 10 \
 --output data/crawled_web.jsonl
```

### 4. Train the Model (with or without on-the-fly crawling)

```bash
# Train on existing crawled data with packed sequences and validation evaluation:
python3 train.py \
 --dataset_path data/crawled_dataset.jsonl \
 --packed_dataset \
 --vocab_size 2048 \
 --block_size 512 \
 --n_layer 6 \
 --n_head 6 \
 --n_embd 384 \
 --batch_size 8 \
 --max_iters 500 \
 --learning_rate 3e-4 \
 --eval_interval 50

# Or crawl and immediately train in a single command:
python3 train.py \
 --crawl_topics "Calculus,Linear Algebra,Classical Mechanics" \
 --packed_dataset \
 --max_iters 300
```

### 5. Start the API Server

```bash
python3 server.py
# Server starts at http://localhost:8000
# Interactive Swagger docs available at: http://localhost:8000/docs
# Interactive Web Testbed available at: http://localhost:8000/
```


---

## API Specification for Purelearn Frontend

### 1. Socratic Chat (`POST /api/chat`)

**Request Body:**
```json
{
 "messages": [
 {"role": "user", "content": "How do I take the derivative of f(x) = x^2 * sin(x)?"}
 ],
 "socratic_level": 2,
 "adhd_mode": true,
 "dyslexia_mode": false,
 "bionic_reading": false,
 "temperature": 0.7,
 "stream": false
}
```

**Response:**
```json
{
 "response": "**Focus Step:** Let's look at this step-by-step: f(x) = x^2 \\sin(x)\n\n**Question to Consider:** Notice that this function is made of two separate functions multiplied together. What are those two distinct parts?",
 "raw_response": "<|adhd_chunk|> Let's look at this step-by-step: <|katex_math|> f(x) = x^2 \\sin(x) <|socratic_question|> Notice that this function is made of two separate functions multiplied together. What are those two distinct parts?",
 "tokens_generated": 48,
 "socratic_level": 2,
 "adhd_mode": true,
 "dyslexia_mode": false
}
```

> **Streaming (SSE):** Set `"stream": true` to receive real-time token chunks via Server-Sent Events (`text/event-stream`).

---

### 2. Synthesize Study Notes (`POST /api/study-notes`)

**Request Body:**
```json
{
 "content": "Calculus Product Rule derivation and examples",
 "adhd_mode": true
}
```

**Response:**
```json
{
 "topic": "Calculus Product Rule derivation and examples",
 "notes_markdown": "# Study Notes: Calculus Product Rule\n\n## Core Concepts\n- **Product Rule:** $\\frac{d}{dx}[u \\cdot v] = u'v + uv'$\n..."
}
```

---

### 3. Generate Active Recall Flashcards (`POST /api/flashcards`)

**Request Body:**
```json
{
 "content": "Newton's Laws of Motion",
 "num_cards": 5
}
```

**Response:**
```json
{
 "flashcards": [
 {
 "question": "What is Newton's Second Law of Motion?",
 "answer": "$F = m \\cdot a$ (Force equals mass times acceleration)",
 "category": "Physics - Mechanics"
 }
 ]
}
```

---

### 4. Reformat Text (`POST /api/adapt-formatting`)

**Request Body:**
```json
{
 "text": "Photosynthesis is the process by which green plants transform light energy into chemical energy.",
 "adhd_mode": true,
 "dyslexia_mode": false,
 "bionic_reading": true
}
```

---

## Integration with `purelearn.vigilance.rw`

In your frontend application (e.g. React/Vite in `purelearn.vigilance.rw`), configure your API client to point to this service:

```typescript
// Purelearn API Client Configuration
const PURELEARN_AI_ENDPOINT = "https://your-purelearn-ai-host.com/api";

export async function askSocraticTutor(messages, settings) {
 const response = await fetch(`${PURELEARN_AI_ENDPOINT}/chat`, {
 method: "POST",
 headers: { "Content-Type": "application/json" },
 body: JSON.stringify({
 messages,
 socratic_level: settings.socraticLevel ?? 2,
 adhd_mode: settings.adhdMode ?? false,
 dyslexia_mode: settings.dyslexiaMode ?? false,
 bionic_reading: settings.bionicReading ?? false,
 temperature: 0.7,
 stream: false
 })
 });
 return await response.json();
}
```