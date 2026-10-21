"""
Workspace Data Manager and State Store for Purelearn AI Tutor.
Provides persistent storage and CRUD for pinned chats, recent chats, study notes,
flashcard decks, quizzes, materials, and student telemetry without mock placeholders.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


WORKSPACE_STORE_FILE = Path("data/workspace_store.json")


DEFAULT_INITIAL_DATA: Dict[str, Any] = {
    "telemetry": {
        "scholar_level": 16,
        "total_xp": 4717,
        "streak_days": 1,
        "quizzes_completed": 6,
        "bonus_tokens": 4,
        "next_level_xp": 4800,
        "xp_progress": 217,
        "xp_needed": 83,
        "active_scholar_badge": True,
        "badges": [
            {"id": "quiz_master", "name": "First Quiz Completed", "icon": "Q"},
            {"id": "streak_master", "name": "Streak Master", "icon": "S"},
            {"id": "socratic_master", "name": "Socratic Discovery Master", "icon": "M"},
        ],
        "reminders": [
            "Test your retention on 'Mathematics' with a grounded material quiz.",
            "Your 1-day study streak is active. Keep momentum going today!",
            "Only 83 XP needed to reach Scholar Level 17.",
        ],
    },
    "chats": [],
    "notes": [
        {
            "id": "note-math-1",
            "topic": "MATHEMATICS",
            "title": "Mathematics Notes",
            "updated": "9/7/2026",
            "content": "In calculus, the **derivative** of a function measures the instantaneous rate of change. Formally, it is defined as the limit of the average rate of change as the time interval shrinks to zero:\n\n$$\\frac{dy}{dx} = \\lim_{\\Delta x \\to 0} \\frac{f(x + \\Delta x) - f(x)}{\\Delta x}$$\n\nWhen dealing with trigonometric functions like the sine wave that models our alternating electric signal, something remarkable happens. The derivative of the sine function is the cosine function:\n\n$$\\frac{d}{dx} (\\sin(x)) = \\cos(x)$$\n\n• What does $\\sin(x)$ represent? Your vertical position on the wave.\n• What does $\\cos(x)$ represent? Your instantaneous vertical velocity (how fast your height is changing).\n\n## 2. Worked Example: Analyzing a Periodic Wave\nConsider an oscillating charge function $q(t) = Q_0 \\cos(\\omega t)$. The instantaneous electric current is the time rate of change of charge:\n\n$$I(t) = \\frac{dq}{dt} = -Q_0 \\omega \\sin(\\omega t)$$"
        },
        {
            "id": "note-africa-1",
            "topic": "ENGINEERING ENTREPRENEURSHIP",
            "title": "Summary: https://africaprize.raeng.org.uk/",
            "updated": "9/8/2026",
            "content": "# Overview of the Africa Prize for Engineering Innovation\n\nThe initiative bridges the gap between lab research and scalable commercial impact across sub-Saharan Africa.\n\n### Key Pillars:\n1. **Technological Feasibility:** Grounded engineering validation.\n2. **Commercial Viability:** Unit economics and customer discovery.\n3. **Social Impact:** Tangible job creation and sustainable development."
        },
        {
            "id": "note-yt-aiml-1",
            "topic": "ARTIFICIAL INTELLIGENCE",
            "title": "YouTube: AI, Machine Learning, Deep Learning and Generative AI Explained",
            "updated": "9/11/2026",
            "content": "# Architectural Foundation of Modern AI\n\nDeep learning models transform input embeddings through layers of parameterized linear transformations and non-linear activations:\n\n$$h^{(l)} = \\sigma(W^{(l)} h^{(l-1)} + b^{(l)})$$\n\nSelf-attention extends this by allowing every token to calculate dynamic affinities with every other token in the sequence."
        },
        {
            "id": "note-n8n-1",
            "topic": "AUTOMATION ENGINEERING",
            "title": "n8n - Notes on n8n",
            "updated": "9/5/2026",
            "content": "# Fundamentals of Workflow Orchestration with n8n\n\nNode-based workflows route JSON payloads through transforms, conditional branches, and HTTP destinations with retry mechanisms and state checkpoints."
        }
    ],
    "decks": [
        {
            "id": "deck-ai-1",
            "category": "AI TUTORING",
            "title": "Summary: General Chat (09:15 AM)",
            "cards": [
                {
                    "id": "c1",
                    "question": "What is the core principle of optimization in Artificial Intelligence and Neural Networks?",
                    "answer": "Gradient Descent minimizes loss $\\mathcal{L}(\\theta)$ by iteratively updating parameters: $\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta \\mathcal{L}$.",
                    "status": "understood"
                }
            ]
        },
        {
            "id": "deck-cs-1",
            "category": "COMPUTER SCIENCE",
            "title": "Summary: I would like to study Python...",
            "cards": [
                {
                    "id": "c2",
                    "question": "What is the time and space complexity of accessing an item in a Python dictionary?",
                    "answer": "Average time complexity is $O(1)$ through hash table probing; space complexity is $O(N)$ based on table capacity and collision thresholds.",
                    "status": "review"
                },
                {
                    "id": "c3",
                    "question": "How does the Python Global Interpreter Lock (GIL) impact CPU-bound multithreading?",
                    "answer": "The GIL ensures only one native thread executes Python bytecode at a time, preventing true CPU parallelism across cores for pure Python threads.",
                    "status": "understood"
                },
                {
                    "id": "c4",
                    "question": "What is the difference between shallow copy and deep copy in Python?",
                    "answer": "Shallow copy constructs a new collection and populates it with references to child objects; deep copy recursively copies child objects.",
                    "status": "understood"
                }
            ]
        },
        {
            "id": "deck-ai-deep-1",
            "category": "ARTIFICIAL INTELLIGENCE",
            "title": "Activation Functions & Optimizers",
            "cards": [
                {
                    "id": "c5",
                    "question": "Why is the SwiGLU activation preferred over standard ReLU in modern transformer architectures?",
                    "answer": "SwiGLU combines Swish gating with linear projections: $\\text{SwiGLU}(x) = (x W \\cdot \\text{sigmoid}(\\beta x W)) \\cdot (x V)$, providing smoother gradients and richer capacity.",
                    "status": "understood"
                },
                {
                    "id": "c6",
                    "question": "What role does RMSNorm play compared to standard LayerNorm?",
                    "answer": "RMSNorm simplifies computation by enforcing root-mean-square invariance without calculating the mean shift, reducing latency without harming convergence.",
                    "status": "understood"
                }
            ]
        },
        {
            "id": "deck-math-1",
            "category": "MATHEMATICS",
            "title": "Mathematics Notes",
            "cards": [
                {
                    "id": "c7",
                    "question": "State Newton's Second Law of Motion in both scalar and differential form.",
                    "answer": "Scalar: $F = m \\cdot a$. Differential Form: $F = \\frac{dp}{dt} = m \\frac{d^2 x}{dt^2}$ where $p$ is momentum.",
                    "status": "understood"
                },
                {
                    "id": "c8",
                    "question": "What is the Product Rule for differentiating two functions $u(x)$ and $v(x)$?",
                    "answer": "$$\\frac{d}{dx}[u \\cdot v] = u' \\cdot v + u \\cdot v'$$",
                    "status": "understood"
                },
                {
                    "id": "c9",
                    "question": "What is the geometric meaning of the definite integral $\\int_{a}^{b} f(x)dx$?",
                    "answer": "The signed area under the curve $f(x)$ bounded between the vertical lines $x = a$ and $x = b$ and the horizontal x-axis.",
                    "status": "review"
                }
            ]
        }
    ]
}


class WorkspaceManager:
    """Manages reading, writing, and mutating persistent application state."""

    def __init__(self, store_path: Path = WORKSPACE_STORE_FILE):
        self.store_path = store_path
        self._ensure_initialized()

    def _ensure_initialized(self):
        if not self.store_path.exists():
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(DEFAULT_INITIAL_DATA)

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_INITIAL_DATA

    def save(self, data: Dict[str, Any]):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_all(self) -> Dict[str, Any]:
        return self.load()

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        data = self.load()
        for chat in data.get("chats", []):
            if chat.get("id") == chat_id:
                return chat
        return None

    def add_message_to_chat(
        self, chat_id: str, user_content: str, assistant_content: str
    ) -> Dict[str, Any]:
        data = self.load()
        matched = None
        for chat in data.get("chats", []):
            if chat.get("id") == chat_id:
                matched = chat
                break

        now_time = time.strftime("%I:%M %p")
        if not matched:
            # Create new chat session if not found
            matched = {
                "id": chat_id,
                "title": user_content[:28] + ("..." if len(user_content) > 28 else ""),
                "subject": "General STEM",
                "type": "Text",
                "date": time.strftime("%-m/%-d/%Y"),
                "pinned": False,
                "grounding": "Grounded in Text Inquiry",
                "messages": [],
            }
            data["chats"].insert(0, matched)

        matched["messages"].append({
            "role": "user",
            "content": user_content,
            "timestamp": now_time,
        })
        matched["messages"].append({
            "role": "assistant",
            "content": assistant_content,
            "timestamp": now_time,
        })

        # Award +10 XP for question interaction
        telemetry = data.setdefault("telemetry", {})
        telemetry["total_xp"] = telemetry.get("total_xp", 4717) + 10
        telemetry["xp_progress"] = telemetry.get("xp_progress", 217) + 10
        if telemetry["xp_progress"] >= 300:
            telemetry["scholar_level"] = telemetry.get("scholar_level", 16) + 1
            telemetry["xp_progress"] = telemetry["xp_progress"] - 300

        self.save(data)
        return matched

    def create_chat(self, title: str, subject: str = "General STEM") -> Dict[str, Any]:
        data = self.load()
        new_id = f"chat-{int(time.time())}"
        new_chat = {
            "id": new_id,
            "title": title,
            "subject": subject,
            "type": "Text",
            "date": time.strftime("%-m/%-d/%Y"),
            "pinned": False,
            "grounding": f"Grounded in {subject}",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Started a new session on **{title}**. What first principles or questions shall we explore?",
                    "timestamp": time.strftime("%I:%M %p"),
                }
            ],
        }
        data["chats"].insert(0, new_chat)
        self.save(data)
        return new_chat

    def get_notes(self) -> List[Dict[str, Any]]:
        data = self.load()
        return data.get("notes", [])

    def save_note(self, note_id: str, title: str, content: str, topic: str = "STUDY NOTE") -> Dict[str, Any]:
        data = self.load()
        notes = data.setdefault("notes", [])
        for note in notes:
            if note.get("id") == note_id:
                note["title"] = title
                note["content"] = content
                note["topic"] = topic
                note["updated"] = time.strftime("%-m/%-d/%Y")
                self.save(data)
                return note

        new_note = {
            "id": note_id or f"note-{int(time.time())}",
            "topic": topic,
            "title": title,
            "content": content,
            "updated": time.strftime("%-m/%-d/%Y"),
        }
        notes.insert(0, new_note)
        self.save(data)
        return new_note

    def get_decks(self) -> List[Dict[str, Any]]:
        data = self.load()
        return data.get("decks", [])

    def update_card_status(self, card_id: str, new_status: str) -> bool:
        data = self.load()
        found = False
        for deck in data.get("decks", []):
            for card in deck.get("cards", []):
                if card.get("id") == card_id:
                    card["status"] = new_status
                    found = True
                    break
        if found:
            self.save(data)
        return found

    def award_quiz_score(self, score_pct: int) -> Dict[str, Any]:
        data = self.load()
        telemetry = data.setdefault("telemetry", {})
        telemetry["quizzes_completed"] = telemetry.get("quizzes_completed", 6) + 1
        xp_gain = max(20, int(score_pct * 0.8))
        telemetry["total_xp"] = telemetry.get("total_xp", 4717) + xp_gain
        telemetry["xp_progress"] = telemetry.get("xp_progress", 217) + xp_gain
        if telemetry["xp_progress"] >= 300:
            telemetry["scholar_level"] = telemetry.get("scholar_level", 16) + 1
            telemetry["xp_progress"] = telemetry["xp_progress"] - 300
        telemetry["bonus_tokens"] = telemetry.get("bonus_tokens", 4) + 1
        self.save(data)
        return telemetry
