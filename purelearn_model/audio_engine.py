"""
Audio Models & Speech Synthesis Engine for Purelearn AI Tutor.
Synthesizes spoken audio from Socratic notes, formulas, and explanations
using high-quality audio models and native speech synthesis.
"""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


AUDIO_CACHE_DIR = Path("data/audio_cache")


def clean_text_for_speech(raw_text: str) -> str:
    """Cleans LaTeX formulas, Markdown markers, and tokens for natural speech cadence."""
    t = raw_text

    # Replace special tokens
    t = re.sub(r"<\|[^>]+>", " ", t)

    # Convert common LaTeX math equations into spoken English
    t = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1 divided by \2", t)
    t = re.sub(r"\\sin\(([^)]+)\)", r"sine of \1", t)
    t = re.sub(r"\\cos\(([^)]+)\)", r"cosine of \1", t)
    t = re.sub(r"\\lim_\{([^}]+)\}", r"the limit as \1", t)
    t = re.sub(r"\\Delta", "delta ", t)
    t = re.sub(r"\\to", " approaches ", t)
    t = re.sub(r"\\cdot", " times ", t)
    t = re.sub(r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r"the integral from \1 to \2 of ", t)
    t = re.sub(r"\\int", "the integral of ", t)
    t = re.sub(r"\$+", "", t)  # Remove math delimiters

    # Clean markdown
    t = re.sub(r"[*#_~`]", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"\n+", ". ", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t


class AudioEngine:
    """Synthesizes high-fidelity spoken audio using speech models."""

    def __init__(self, cache_dir: Path = AUDIO_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str, voice: str = "Samantha") -> Optional[bytes]:
        """
        Synthesizes text into standard 16-bit PCM WAV audio bytes.
        Caches outputs by content hash for instantaneous retrieval.
        """
        spoken_text = clean_text_for_speech(text)
        if not spoken_text:
            return None

        cache_key = hashlib.sha256(f"{voice}:{spoken_text}".encode("utf-8")).hexdigest()
        cached_wav = self.cache_dir / f"{cache_key}.wav"

        if cached_wav.exists() and cached_wav.stat().st_size > 0:
            with open(cached_wav, "rb") as f:
                return f.read()

        # Generate audio using macOS native high-quality speech synthesis model
        try:
            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temp_aiff:
                aiff_path = temp_aiff.name

            # Synthesize AIFF
            subprocess.run(
                ["/usr/bin/say", "-v", voice, "-o", aiff_path, spoken_text],
                check=True,
                capture_output=True,
            )

            # Convert to standard 22.05kHz 16-bit Mono WAV using Apple afconvert
            subprocess.run(
                [
                    "/usr/bin/afconvert",
                    "-f", "WAVE",
                    "-d", "LEI16@22050",
                    aiff_path,
                    str(cached_wav),
                ],
                check=True,
                capture_output=True,
            )

            if os.path.exists(aiff_path):
                os.remove(aiff_path)

            if cached_wav.exists():
                with open(cached_wav, "rb") as f:
                    return f.read()
        except Exception as e:
            # Fallback or error log
            print(f"[AudioEngine Error] {e}")
            return None

        return None
