"""
Purelearn AI Model FastAPI Server & API Wrapper.
Exposes REST and SSE streaming endpoints for direct integration with purelearn.vigilance.rw.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from purelearn_model.config import PureLearnConfig, TokenizerConfig
from purelearn_model.engine import PureLearnEngine
from purelearn_model.model import PureLearnLM
from purelearn_model.tokenizer import PureLearnTokenizer

app = FastAPI(
    title="Purelearn AI Engine API",
    description="Custom PyTorch Neuro-Inclusive Socratic AI Model and Educational API for Purelearn (purelearn.vigilance.rw)",
    version="1.0.0",
)

# Enable CORS for purelearn frontend domains and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://purelearn.vigilance.rw",
        "https://purelearn.ai",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Tokenizer, Model, and Engine
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / "purelearn_latest.pt"

if TOKENIZER_PATH.exists():
    tokenizer = PureLearnTokenizer.load(TOKENIZER_PATH)
else:
    tokenizer = PureLearnTokenizer()

model_config = PureLearnConfig(vocab_size=tokenizer.vocab_size)

if CHECKPOINT_PATH.exists():
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            model_config = checkpoint["config"]
        model = PureLearnLM(model_config)
        engine = PureLearnEngine(model=model, tokenizer=tokenizer)
        engine.load_checkpoint(str(CHECKPOINT_PATH))
        print(f"[SUCCESS] Loaded PureLearn model checkpoint ({model.get_num_params():,} parameters) from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint ({e}), falling back to fresh model.")
        model = PureLearnLM(model_config)
        engine = PureLearnEngine(model=model, tokenizer=tokenizer)
else:
    model = PureLearnLM(model_config)
    engine = PureLearnEngine(model=model, tokenizer=tokenizer)


# Pydantic Request Models
class Message(BaseModel):
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Conversation history")
    socratic_level: int = Field(2, ge=1, le=3, description="Socratic depth: 1 (Gentle hints), 2 (Guided questions), 3 (Pure discovery)")
    adhd_mode: bool = Field(False, description="Enable ADHD micro-chunking & focus highlighting")
    dyslexia_mode: bool = Field(False, description="Enable Dyslexia high visual clarity & simplified syntax")
    bionic_reading: bool = Field(False, description="Apply bionic fixation bolding to response")
    stream: bool = Field(False, description="Stream response tokens via Server-Sent Events (SSE)")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(350, ge=10, le=1024)


class StudyNotesRequest(BaseModel):
    content: str = Field(..., description="Topic or chat transcript to synthesize into study notes")
    adhd_mode: bool = Field(False, description="Format notes with high-contrast bulleted micro-chunks")


class FlashcardsRequest(BaseModel):
    content: str = Field(..., description="Topic or chat transcript from which to extract flashcards")
    num_cards: int = Field(5, ge=1, le=20)


class AdaptFormattingRequest(BaseModel):
    text: str = Field(..., description="Educational text to reformat")
    adhd_mode: bool = Field(False)
    dyslexia_mode: bool = Field(False)
    bionic_reading: bool = Field(False)


# API Endpoints
@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": "Purelearn AI Backend",
        "device": engine.device,
        "parameters": model.get_num_params(),
        "timestamp": time.time(),
    }


@app.get("/api/model/info")
def get_model_info():
    """Returns model architecture parameters and configuration."""
    return {
        "model_name": "PureLearn-Transformer-PyTorch",
        "device": engine.device,
        "vocab_size": tokenizer.vocab_size,
        "block_size": model.config.block_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "total_parameters": model.get_num_params(),
        "special_tokens": tokenizer.special_tokens,
    }


@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """
    Socratic neuro-inclusive chat endpoint.
    Supports both standard JSON response and SSE streaming.
    """
    msg_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        def event_generator():
            for piece in engine.chat_stream(
                messages=msg_dicts,
                socratic_level=request.socratic_level,
                adhd_mode=request.adhd_mode,
                dyslexia_mode=request.dyslexia_mode,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
            ):
                payload = json.dumps({"token": piece})
                yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    result = engine.chat(
        messages=msg_dicts,
        socratic_level=request.socratic_level,
        adhd_mode=request.adhd_mode,
        dyslexia_mode=request.dyslexia_mode,
        bionic_reading=request.bionic_reading,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_new_tokens,
    )
    return result


@app.post("/api/study-notes")
def generate_study_notes_endpoint(request: StudyNotesRequest):
    """Generates structured Markdown study notes with KaTeX math equations."""
    notes = engine.generate_study_notes(request.content)
    return notes


@app.post("/api/flashcards")
def generate_flashcards_endpoint(request: FlashcardsRequest):
    """Generates active recall flashcards with question, answer, and category."""
    cards = engine.generate_flashcards(request.content)
    return {"flashcards": cards[: request.num_cards]}


@app.post("/api/adapt-formatting")
def adapt_formatting_endpoint(request: AdaptFormattingRequest):
    """Dynamically reformats text for ADHD / Dyslexia readability."""
    formatted = engine.format_neuro_inclusive(
        request.text,
        adhd_mode=request.adhd_mode,
        dyslexia_mode=request.dyslexia_mode,
        bionic_reading=request.bionic_reading,
    )
    return {"formatted_text": formatted}


@app.get("/", response_class=HTMLResponse)
def root_view():
    """Serves the interactive web test client."""
    demo_file = Path("demo_client.html")
    if demo_file.exists():
        return HTMLResponse(content=demo_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Purelearn AI PyTorch Service is Running. Visit /api/health or /docs for API schema.</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
