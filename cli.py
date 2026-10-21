#!/usr/bin/env python3
"""
PureLearn AI - Interactive Terminal Tutor
Socratic Neuro-Inclusive AI Engine with NLP Human Communication & Affect Adaptation.
Runs entirely in the terminal with real-time streaming, KaTeX formatting,
and zero emojis.
"""

import argparse
import os
import readline  # Enables terminal line editing and command history
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from purelearn_model.config import PureLearnConfig
from purelearn_model.dataset import strip_emojis
from purelearn_model.engine import HumanCommunicationAnalyzer, PureLearnEngine
from purelearn_model.model import PureLearnLM
from purelearn_model.tokenizer import PureLearnTokenizer

# Terminal ANSI Color & Style Codes
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


def print_banner(device: str, num_params: int, checkpoint_name: str):
    """Displays clean, non-emoji ASCII header banner."""
    print(f"{CYAN}{BOLD}{'=' * 72}{RESET}")
    print(f"{CYAN}{BOLD} PURELEARN AI - SOCRATIC & NEURO-INCLUSIVE TERMINAL TUTOR{RESET}")
    print(f"{DIM} PyTorch Transformer LM | NLP Human Communication & Affect Engine{RESET}")
    print(f"{CYAN}{'=' * 72}{RESET}")
    print(f" {BOLD}Device:{RESET} {device.upper()}  |  {BOLD}Parameters:{RESET} {num_params:,}  |  {BOLD}Source:{RESET} {checkpoint_name}")
    print(f"{CYAN}{'-' * 72}{RESET}")
    print(f" {BOLD}Interactive Commands:{RESET}")
    print(f"   {GREEN}/level <1|2|3>{RESET}  Set Socratic depth (1: Gentle, 2: Guided, 3: Strict)")
    print(f"   {GREEN}/adhd{RESET}           Toggle ADHD micro-chunking mode")
    print(f"   {GREEN}/dyslexia{RESET}       Toggle Dyslexia high-readability mode")
    print(f"   {GREEN}/bionic{RESET}         Toggle Bionic reading fixations")
    print(f"   {GREEN}/nlp{RESET}            Toggle real-time NLP affect & intent inspection")
    print(f"   {GREEN}/notes <topic>{RESET}  Synthesize structured Markdown study notes")
    print(f"   {GREEN}/cards <topic>{RESET}  Generate active recall flashcards")
    print(f"   {GREEN}/clear{RESET}          Clear active conversation history")
    print(f"   {GREEN}/help{RESET}           Show this command cheat-sheet")
    print(f"   {GREEN}/exit{RESET} or {GREEN}/quit{RESET}   Exit terminal session")
    print(f"{CYAN}{'=' * 72}{RESET}\n")


def display_nlp_profile(profile: Dict):
    """Renders human communication NLP analysis in terminal."""
    affect = profile.get("primary_affect", "neutral").upper()
    verbosity = profile.get("verbosity", "conversational")
    misconception = profile.get("misconception_type") or "None"
    
    tags = []
    if profile.get("is_confused"):
        tags.append(f"{YELLOW}CONFUSED{RESET}")
    if profile.get("is_frustrated"):
        tags.append(f"{RED}FRUSTRATED{RESET}")
    if profile.get("is_tentative"):
        tags.append(f"{MAGENTA}TENTATIVE{RESET}")
    if profile.get("is_curious"):
        tags.append(f"{CYAN}CURIOUS{RESET}")
    if profile.get("is_confident"):
        tags.append(f"{GREEN}CONFIDENT{RESET}")
    if profile.get("is_analogy"):
        tags.append(f"{BLUE}ANALOGY-SEEKING{RESET}")

    tags_str = " | ".join(tags) if tags else f"{DIM}Neutral Inquisitive{RESET}"

    print(f"{DIM}[NLP Adaptation] Affect: {affect} | Signals: {tags_str} | Pacing: {verbosity} | Misconception: {misconception}{RESET}")


def load_engine(checkpoint_arg: Optional[str] = None, device_arg: Optional[str] = None) -> PureLearnEngine:
    """Loads tokenizer, model configuration, and best available weights."""
    checkpoint_dir = Path("checkpoints")
    tokenizer_path = checkpoint_dir / "tokenizer.json"

    if tokenizer_path.exists():
        tokenizer = PureLearnTokenizer.load(str(tokenizer_path))
    else:
        tokenizer = PureLearnTokenizer()

    # Determine checkpoint file
    selected_checkpoint = None
    if checkpoint_arg:
        if Path(checkpoint_arg).exists():
            selected_checkpoint = Path(checkpoint_arg)
        else:
            print(f"{YELLOW}[WARNING] Specified checkpoint '{checkpoint_arg}' not found.{RESET}")

    if selected_checkpoint is None:
        best_ckpt = checkpoint_dir / "purelearn_best.pt"
        latest_ckpt = checkpoint_dir / "purelearn_latest.pt"
        if best_ckpt.exists():
            selected_checkpoint = best_ckpt
        elif latest_ckpt.exists():
            selected_checkpoint = latest_ckpt

    # Configure model
    model_config = PureLearnConfig(vocab_size=tokenizer.vocab_size)
    if selected_checkpoint and selected_checkpoint.exists():
        try:
            ckpt_meta = torch.load(str(selected_checkpoint), map_location="cpu", weights_only=False)
            if isinstance(ckpt_meta, dict) and "config" in ckpt_meta:
                model_config = ckpt_meta["config"]
        except Exception:
            pass

    model = PureLearnLM(model_config)
    engine = PureLearnEngine(model=model, tokenizer=tokenizer, device=device_arg)

    if selected_checkpoint and selected_checkpoint.exists():
        try:
            engine.load_checkpoint(str(selected_checkpoint))
            checkpoint_name = selected_checkpoint.name
        except Exception as e:
            checkpoint_name = f"Fresh Init (Load error: {e})"
    else:
        checkpoint_name = "Fresh Initialization (No weights found)"

    engine._loaded_name = checkpoint_name
    return engine


def run_single_shot(engine: PureLearnEngine, query: str, args):
    """Executes a single non-interactive prompt and prints output."""
    messages = [{"role": "user", "content": query}]
    if args.show_nlp:
        profile = engine.nlp_analyzer.analyze(query, messages)
        display_nlp_profile(profile)

    res = engine.chat(
        messages=messages,
        socratic_level=args.socratic_level,
        adhd_mode=args.adhd,
        dyslexia_mode=args.dyslexia,
        bionic_reading=args.bionic,
        temperature=args.temp,
    )
    print(f"\n{BOLD}Purelearn AI:{RESET}\n")
    print(res["response"])
    print()


def run_interactive(engine: PureLearnEngine, args):
    """Runs the main interactive conversational REPL."""
    print_banner(
        device=engine.device,
        num_params=engine.model.get_num_params(),
        checkpoint_name=engine._loaded_name,
    )

    history: List[Dict[str, str]] = []
    socratic_level = args.socratic_level
    adhd_mode = args.adhd
    dyslexia_mode = args.dyslexia
    bionic_reading = args.bionic
    show_nlp = args.show_nlp
    temperature = args.temp

    level_names = {1: "Gentle Hints", 2: "Guided Discovery", 3: "Strict Inquiry"}

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}Student > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{CYAN}Session ended. Thank you for learning with Purelearn AI!{RESET}\n")
            break

        if not user_input:
            continue

        # Handle Slash Commands
        lower_input = user_input.lower()
        if lower_input in ["/exit", "/quit", "exit", "quit", ":q"]:
            print(f"\n{CYAN}Session ended. Thank you for learning with Purelearn AI!{RESET}\n")
            break

        if lower_input == "/clear":
            history = []
            print(f"{YELLOW}[INFO] Dialogue history reset.{RESET}\n")
            continue

        if lower_input == "/help":
            print(f"\n{CYAN}Available Commands:{RESET}")
            print(f"  /level <1-3>   Current: Level {socratic_level} ({level_names.get(socratic_level)})")
            print(f"  /adhd          Current: {'Enabled' if adhd_mode else 'Disabled'}")
            print(f"  /dyslexia      Current: {'Enabled' if dyslexia_mode else 'Disabled'}")
            print(f"  /bionic        Current: {'Enabled' if bionic_reading else 'Disabled'}")
            print(f"  /nlp           Current: {'Enabled' if show_nlp else 'Disabled'}")
            print("  /notes <topic> Generate Markdown study notes")
            print("  /cards <topic> Generate active recall flashcards")
            print("  /clear         Reset conversation")
            print("  /exit          Quit\n")
            continue

        if lower_input.startswith("/level"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1] in ["1", "2", "3"]:
                socratic_level = int(parts[1])
                print(f"{GREEN}[SETTING] Socratic Depth set to Level {socratic_level} ({level_names[socratic_level]}).{RESET}\n")
            else:
                print(f"{YELLOW}[USAGE] /level <1|2|3> (1: Gentle, 2: Guided, 3: Strict){RESET}\n")
            continue

        if lower_input == "/adhd":
            adhd_mode = not adhd_mode
            status = f"{GREEN}ENABLED{RESET}" if adhd_mode else f"{YELLOW}DISABLED{RESET}"
            print(f"[SETTING] ADHD Micro-Chunking is now {status}.\n")
            continue

        if lower_input == "/dyslexia":
            dyslexia_mode = not dyslexia_mode
            status = f"{GREEN}ENABLED{RESET}" if dyslexia_mode else f"{YELLOW}DISABLED{RESET}"
            print(f"[SETTING] Dyslexia Readability Mode is now {status}.\n")
            continue

        if lower_input == "/bionic":
            bionic_reading = not bionic_reading
            status = f"{GREEN}ENABLED{RESET}" if bionic_reading else f"{YELLOW}DISABLED{RESET}"
            print(f"[SETTING] Bionic Reading Fixations is now {status}.\n")
            continue

        if lower_input == "/nlp":
            show_nlp = not show_nlp
            status = f"{GREEN}ENABLED{RESET}" if show_nlp else f"{YELLOW}DISABLED{RESET}"
            print(f"[SETTING] Real-time NLP Affect Analysis is now {status}.\n")
            continue

        if lower_input.startswith("/notes"):
            topic = user_input[6:].strip()
            if not topic:
                topic = history[-1]["content"] if history else "STEM Core Principles"
            print(f"\n{CYAN}{BOLD}--- Synthesizing Study Notes: {topic} ---{RESET}\n")
            notes = engine.generate_study_notes(topic, adhd_mode=adhd_mode)
            print(notes.get("notes_markdown", ""))
            print(f"\n{CYAN}{'-' * 72}{RESET}\n")
            continue

        if lower_input.startswith("/cards"):
            topic = user_input[6:].strip()
            if not topic:
                topic = history[-1]["content"] if history else "Key STEM Rules"
            print(f"\n{CYAN}{BOLD}--- Active Recall Flashcards: {topic} ---{RESET}\n")
            cards = engine.generate_flashcards(topic, num_cards=3)
            for idx, c in enumerate(cards, 1):
                print(f"{BOLD}Card {idx} [{c.get('category', 'Concept')}]{RESET}")
                print(f"  {YELLOW}Q:{RESET} {c.get('question')}")
                print(f"  {GREEN}A:{RESET} {c.get('answer')}\n")
            print(f"{CYAN}{'-' * 72}{RESET}\n")
            continue

        # Process user dialogue turn
        clean_user = strip_emojis(user_input)
        history.append({"role": "user", "content": clean_user})

        # NLP Human Affect Analysis
        profile = engine.nlp_analyzer.analyze(clean_user, history)
        if show_nlp:
            display_nlp_profile(profile)

        # Generate Socratic Response
        print(f"\n{CYAN}{BOLD}Purelearn AI > {RESET}", end="", flush=True)

        res = engine.chat(
            messages=history,
            socratic_level=socratic_level,
            adhd_mode=adhd_mode,
            dyslexia_mode=dyslexia_mode,
            bionic_reading=bionic_reading,
            temperature=temperature,
        )

        response_text = res["response"]

        # Stream words out for responsive human-like pacing
        words = response_text.split(" ")
        for i, word in enumerate(words):
            end_char = " " if i < len(words) - 1 else "\n"
            sys.stdout.write(word + end_char)
            sys.stdout.flush()
            time.sleep(0.015)  # Smooth fluid typing cadence

        history.append({"role": "assistant", "content": response_text})
        print()


def main():
    parser = argparse.ArgumentParser(
        description="PureLearn AI: Socratic Neuro-Inclusive Terminal Tutor"
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Optional single prompt to evaluate directly")
    parser.add_argument("-q", "--query", type=str, default=None, help="Direct inquiry query text")
    parser.add_argument("-i", "--interactive", action="store_true", help="Force interactive REPL mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint (.pt)")
    parser.add_argument("--device", type=str, default=None, help="Device to execute on ('cuda', 'mps', 'cpu')")
    parser.add_argument("--socratic-level", type=int, default=2, choices=[1, 2, 3], help="1=Gentle, 2=Guided, 3=Strict")
    parser.add_argument("--adhd", action="store_true", help="Enable ADHD micro-chunking and visual focus")
    parser.add_argument("--dyslexia", action="store_true", help="Enable dyslexia high-readability formatting")
    parser.add_argument("--bionic", action="store_true", help="Enable bionic reading fixations")
    parser.add_argument("--show-nlp", action="store_true", help="Display real-time NLP intent and affect analysis")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    engine = load_engine(checkpoint_arg=args.checkpoint, device_arg=args.device)

    query_text = args.query or args.prompt
    if query_text and not args.interactive:
        run_single_shot(engine, query_text, args)
    else:
        run_interactive(engine, args)


if __name__ == "__main__":
    main()
