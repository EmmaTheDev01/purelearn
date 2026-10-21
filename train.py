"""
Enhanced Training, Internet Ingestion & Scaling Script for PureLearn PyTorch Transformer Model.

Features:
- Internet data crawling & ingestion pipeline integration (--crawl, --crawl_topics, --crawl_stem_pack, --crawl_urls)
- Dual dataset support: Socratic dialogues and continuous packed pretraining documents
- Validation split, periodic evaluation, and Perplexity metric tracking
- Automatic Mixed Precision (AMP) and Gradient Accumulation for memory-efficient scaling
- Checkpointing: saves latest step, periodic step, and best validation model
- KeyboardInterrupt graceful shutdown and recovery
- Multi-GPU, Apple Silicon (MPS), and CPU auto-detection
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from purelearn_model.config import PureLearnConfig, TokenizerConfig, TrainingConfig
from purelearn_model.crawler import CrawlerConfig, InternetDataPipeline, DEFAULT_STEM_TOPICS
from purelearn_model.dataset import (
    PackedPureLearnDataset,
    PureLearnDataset,
    collate_purelearn_batch,
    create_train_val_split,
    format_dialogue_to_text,
    get_default_curriculum_texts,
)
from purelearn_model.model import PureLearnLM
from purelearn_model.tokenizer import PureLearnTokenizer


def get_lr(step: int, config: TrainingConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_iters:
        return config.learning_rate * step / max(1, config.warmup_iters)
    if step > config.lr_decay_iters:
        return config.min_learning_rate
    decay_ratio = (step - config.warmup_iters) / max(1, (config.lr_decay_iters - config.warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def load_dataset_texts(dataset_path: str, tokenizer: PureLearnTokenizer) -> List[str]:
    """
    Loads training texts from a JSON/JSONL/TXT dataset file or directory.
    Handles dialogue conversations, pretraining articles, and raw educational text.
    """
    path = Path(dataset_path)
    if not path.exists():
        print(f"[WARNING] Dataset path '{dataset_path}' not found. Using default curriculum texts.")
        return get_default_curriculum_texts(tokenizer)

    texts: List[str] = []
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.jsonl")) + list(path.glob("*.json")) + list(path.glob("*.txt"))

    for f in files:
        if f.suffix == ".jsonl":
            with open(f, "r", encoding="utf-8") as file:
                for line in file:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        item = json.loads(line_str)
                        if isinstance(item, dict):
                            if "dialogue" in item:
                                formatted = format_dialogue_to_text(item["dialogue"], tokenizer)
                                texts.append(formatted)
                            elif "text" in item:
                                texts.append(item["text"])
                            elif "content" in item:
                                texts.append(item["content"])
                    except Exception:
                        continue
        elif f.suffix == ".json":
            with open(f, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                if "dialogue" in item:
                                    formatted = format_dialogue_to_text(item["dialogue"], tokenizer)
                                    texts.append(formatted)
                                elif "text" in item:
                                    texts.append(item["text"])
                            elif isinstance(item, str):
                                texts.append(item)
                except Exception as e:
                    print(f"[WARNING] Error reading JSON {f}: {e}")
        elif f.suffix == ".txt":
            with open(f, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    texts.append(content)

    print(f"[INFO] Loaded {len(texts):,} training text records from '{dataset_path}'")
    return texts if texts else get_default_curriculum_texts(tokenizer)


@torch.no_grad()
def evaluate_model(
    model: PureLearnLM,
    val_loader: DataLoader,
    device: str,
    max_eval_batches: int = 15,
) -> Tuple[float, float]:
    """
    Computes validation cross-entropy loss and Perplexity (exp(loss)) on held-out data.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for x, y in val_loader:
        if total_batches >= max_eval_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss, _ = model(x, targets=y)
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            total_batches += 1

    model.train()
    if total_batches == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_batches
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap exponent to avoid overflow
    return avg_loss, perplexity


def run_pipeline_crawler(args) -> str:
    """Executes the internet crawler before training if requested."""
    print("[INFO] Initiating Internet Web Crawling for training data...")
    pipeline = InternetDataPipeline()

    topics = []
    if args.crawl_stem_pack:
        topics.extend(DEFAULT_STEM_TOPICS)
    if args.crawl_topics:
        topics.extend([t.strip() for t in args.crawl_topics.split(",") if t.strip()])

    articles = []
    if topics:
        print(f"  Harvesting {len(topics)} topics from Wikipedia and arXiv...")
        topic_articles = pipeline.harvest_topics(topics, max_per_topic=args.max_per_topic)
        articles.extend(topic_articles)

    if args.crawl_urls:
        urls = [u.strip() for u in args.crawl_urls.split(",") if u.strip()]
        print(f"  Crawling custom seed URLs: {urls}...")
        url_articles = pipeline.harvest_urls(urls, max_pages_per_url=10)
        articles.extend(url_articles)

    if articles:
        target_path = args.crawl_output or "data/crawled_dataset.jsonl"
        count = pipeline.process_and_export(articles, output_file=target_path, generate_socratic=True)
        print(f"[SUCCESS] Harvested {len(articles)} internet materials -> {count} training records saved to '{target_path}'.")
        return target_path

    print("[WARNING] No topics or URLs crawled; proceeding with existing dataset.")
    return args.dataset_path


def train(args):
    # 0. Optional: Crawl Internet Data
    if args.crawl or args.crawl_topics or args.crawl_stem_pack or args.crawl_urls:
        crawled_path = run_pipeline_crawler(args)
        if crawled_path and Path(crawled_path).exists():
            args.dataset_path = crawled_path

    # 1. Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Training PureLearn AI Model on device: {device.upper()}")

    # 2. Tokenizer setup
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = checkpoint_dir / "tokenizer.json"

    if tokenizer_path.exists() and not args.retrain_tokenizer:
        tokenizer = PureLearnTokenizer.load(tokenizer_path)
        print(f"[INFO] Loaded existing tokenizer vocabulary (size: {tokenizer.vocab_size:,})")
    else:
        tokenizer = PureLearnTokenizer()

    # Load dataset texts
    if args.dataset_path:
        all_texts = load_dataset_texts(args.dataset_path, tokenizer)
    else:
        all_texts = get_default_curriculum_texts(tokenizer)

    if not tokenizer_path.exists() or args.retrain_tokenizer:
        tokenizer.train_from_texts(all_texts, target_vocab_size=args.vocab_size)
        tokenizer.save(tokenizer_path)
        print(f"[INFO] Trained & saved new tokenizer vocabulary (size: {tokenizer.vocab_size:,}) to {tokenizer_path}")

    # 3. Model initialization or resumption
    model_config = PureLearnConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    model = PureLearnLM(model_config)

    start_step = 0
    best_val_loss = float("inf")

    if args.resume_from and Path(args.resume_from).exists():
        print(f"[INFO] Resuming model weights from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            start_step = checkpoint.get("step", 0)
            best_val_loss = checkpoint.get("val_loss", checkpoint.get("loss", float("inf")))
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"[INFO] PureLearn Transformer active with {model.get_num_params():,} trainable parameters")

    # 4. Split dataset into Train and Validation
    train_texts, val_texts = create_train_val_split(all_texts, val_ratio=args.val_ratio, seed=42)
    print(f"[INFO] Dataset split: {len(train_texts):,} train sequences | {len(val_texts):,} validation sequences")

    # Choose Dataset type: standard or packed
    if args.packed_dataset:
        print(f"[INFO] Using PackedPureLearnDataset (maximum sequence packing, zero padding waste)")
        train_dataset = PackedPureLearnDataset(train_texts, tokenizer, block_size=args.block_size)
        val_dataset = PackedPureLearnDataset(val_texts, tokenizer, block_size=args.block_size)
        collate_fn = None
    else:
        train_dataset = PureLearnDataset(train_texts, tokenizer, max_seq_len=args.block_size)
        val_dataset = PureLearnDataset(val_texts, tokenizer, max_seq_len=args.block_size)
        collate_fn = lambda b: collate_purelearn_batch(b, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True if len(train_dataset) > args.batch_size else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 5. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    train_config = TrainingConfig(
        learning_rate=args.learning_rate,
        min_learning_rate=args.learning_rate * 0.1,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.max_iters,
        max_iters=args.max_iters,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Automatic Mixed Precision setup
    use_amp = args.use_amp and device in ["cuda", "mps"]
    amp_device_type = "cuda" if device == "cuda" else ("cpu" if device == "cpu" else "mps")
    scaler = torch.amp.GradScaler(device=amp_device_type) if (use_amp and device == "cuda") else None
    if use_amp:
        print(f"[INFO] Automatic Mixed Precision (AMP) enabled for accelerated {device.upper()} training")

    grad_accum_steps = max(1, args.gradient_accumulation_steps)
    if grad_accum_steps > 1:
        effective_batch = args.batch_size * grad_accum_steps
        print(f"[INFO] Gradient Accumulation: {grad_accum_steps} steps (Effective batch size: {effective_batch})")

    model.train()
    step = start_step
    target_step = start_step + args.max_iters
    start_time = time.time()
    accum_loss = 0.0

    print(f"[INFO] Starting PureLearn Training: {args.max_iters} iterations (steps {start_step} -> {target_step})...")

    try:
        while step < target_step:
            for x, y in train_loader:
                if step >= target_step:
                    break

                x = x.to(device)
                y = y.to(device)

                # Cosine learning rate scheduling
                lr = get_lr(step, train_config)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Forward pass with optional AMP
                if use_amp and device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        _, loss, _ = model(x, targets=y)
                        scaled_loss = loss / grad_accum_steps
                    scaler.scale(scaled_loss).backward()
                else:
                    _, loss, _ = model(x, targets=y)
                    scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()

                accum_loss += loss.item()

                # Optimization step after accumulating gradients
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == target_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                step += 1

                # Periodic Logging
                if step % args.log_interval == 0 or step == 1 or step == target_step:
                    elapsed = time.time() - start_time
                    steps_per_sec = (step - start_step) / max(0.1, elapsed)
                    current_loss = loss.item()
                    train_ppl = math.exp(min(current_loss, 20.0))
                    print(
                        f"Step [{step:5d}/{target_step:5d}] | "
                        f"Loss: {current_loss:.4f} | "
                        f"PPL: {train_ppl:.2f} | "
                        f"LR: {lr:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

                # Periodic Validation Evaluation
                if step % args.eval_interval == 0 or step == target_step:
                    val_loss, val_ppl = evaluate_model(model, val_loader, device=device, max_eval_batches=args.eval_iters)
                    print(f"  [EVAL @ Step {step}] Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

                    # Check for new best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = checkpoint_dir / "purelearn_best.pt"
                        best_checkpoint = {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": model_config,
                            "val_loss": val_loss,
                            "val_perplexity": val_ppl,
                        }
                        torch.save(best_checkpoint, best_path)
                        print(f"  [SAVED] New Best Model Saved! Val Loss: {val_loss:.4f} -> {best_path}")

                # Periodic Checkpoint Saving
                if step % args.save_interval == 0 or step == target_step:
                    latest_path = checkpoint_dir / "purelearn_latest.pt"
                    step_path = checkpoint_dir / f"purelearn_step_{step}.pt"

                    checkpoint_data = {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model_config,
                        "loss": loss.item(),
                        "best_val_loss": best_val_loss,
                    }
                    torch.save(checkpoint_data, latest_path)
                    torch.save(checkpoint_data, step_path)
                    print(f"[CHECKPOINT] Saved: {latest_path}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user. Saving emergency checkpoint...")
        latest_path = checkpoint_dir / "purelearn_interrupted.pt"
        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "loss": loss.item() if "loss" in locals() else None,
        }, latest_path)
        print(f"[CHECKPOINT] Interrupted checkpoint safely preserved in '{latest_path}'.")

    total_time = time.time() - start_time
    print(f"[SUCCESS] PureLearn Training session concluded in {total_time:.2f}s! Checkpoints saved in '{args.checkpoint_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale & Train PureLearn AI Model on Internet & Socratic Data")

    # Dataset & Web Ingestion parameters
    parser.add_argument("--dataset_path", type=str, default="data/purelearn_curriculum.jsonl", help="Path to JSON/JSONL dataset file or directory")
    parser.add_argument("--crawl", action="store_true", help="Crawl internet materials before training")
    parser.add_argument("--crawl_topics", type=str, default=None, help="Comma-separated topics to crawl (e.g. 'Calculus,Physics,Algorithms')")
    parser.add_argument("--crawl_stem_pack", action="store_true", help="Crawl full STEM and neuro-inclusive knowledge pack")
    parser.add_argument("--crawl_urls", type=str, default=None, help="Comma-separated seed URLs to crawl")
    parser.add_argument("--crawl_output", type=str, default="data/crawled_dataset.jsonl", help="Where to save crawled data")
    parser.add_argument("--max_per_topic", type=int, default=3, help="Max articles per topic to crawl")
    parser.add_argument("--packed_dataset", action="store_true", help="Use contiguous token packing for maximum throughput on crawled text")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (0.0 - 0.5)")

    # Model Architecture parameters
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocabulary size")
    parser.add_argument("--block_size", type=int, default=512, help="Context sequence length")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of Attention heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # Optimization & Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max_iters", type=int, default=300, help="Number of training iterations")
    parser.add_argument("--use_amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)")

    # Intervals & Hardware
    parser.add_argument("--log_interval", type=int, default=25, help="Logging interval (steps)")
    parser.add_argument("--eval_interval", type=int, default=50, help="Validation evaluation interval (steps)")
    parser.add_argument("--eval_iters", type=int, default=10, help="Number of validation batches to evaluate")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint saving interval")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--retrain_tokenizer", action="store_true", help="Retrain tokenizer vocabulary on new data")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'mps', or 'cpu')")

    parsed_args = parser.parse_args()
    train(parsed_args)
