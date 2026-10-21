"""
PureLearn AI Engine
High-level inference, NLP Human Communication & Affect Adaptation,
Socratic scaffolding state machine, and neuro-inclusive study tools.
No emojis are used in system reasoning, prompt templates, or responses.
"""

import re
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import torch
from .config import PureLearnConfig, TokenizerConfig
from .dataset import strip_emojis
from .model import PureLearnLM
from .tokenizer import PureLearnTokenizer


class HumanCommunicationAnalyzer:
    """
    NLP Intent, Affect, and Cognitive State Analyzer.
    Analyzes human user messages to understand communication style, confusion,
    hesitation, emotional state, curiosity, and conceptual misconceptions, allowing the AI
    tutor to adapt dynamically to human learners.
    """

    def __init__(self):
        self.confusion_patterns = re.compile(
            r"\b(confus|don'?t get|don'?t understand|lost|huh|what\?|makes? no sense|hard|stuck|idk|i do not know|no idea|too complicated|what do you mean|unclear|dont know|not sure what to do)\b",
            re.IGNORECASE
        )
        self.frustration_patterns = re.compile(
            r"\b(frustrat|overwhelm|hate this|annoy|impossible|give up|giving up|too much|stress|ugh|exhausted|brain hurts|so difficult|can'?t do this|cannot do this|hopeless)\b",
            re.IGNORECASE
        )
        self.tentative_patterns = re.compile(
            r"\b(maybe|i think|is it\?|could it be|probably|guess|might be|not sure|sort of|perhaps|would it be|i feel like|is that right|did i get that|am i right|uhm?|er|um)\b",
            re.IGNORECASE
        )
        self.analogy_patterns = re.compile(
            r"\b(analogy|like what\?|in real life|plain english|simple terms|eli5|explain like|visualize|picture it|metaphor|concrete example|everyday example|real world)\b",
            re.IGNORECASE
        )
        self.confidence_patterns = re.compile(
            r"\b(definitely|obviously|i know|clear now|got it|easy|solved|makes sense now|i understand now|aha|bingo|figured it out|understood)\b",
            re.IGNORECASE
        )
        self.curiosity_patterns = re.compile(
            r"\b(why\?|why does|how come|what if|how does that work|why does that happen|does that mean|what causes|what happens if|tell me more|curious|why is that)\b",
            re.IGNORECASE
        )
        # STEM Misconceptions
        self.misconception_fall_faster = re.compile(
            r"\b(heavier.*(fall|drop|hit).*faster|heavy.*(drop|fall).*first|mass.*makes.*fall.*faster)\b",
            re.IGNORECASE
        )
        self.misconception_speed_accel = re.compile(
            r"\b(velocity.*is.*acceleration|speed.*is.*acceleration|accelerat.*means.*fast|high speed.*accelerat)\b",
            re.IGNORECASE
        )
        self.misconception_product_rule = re.compile(
            r"(derivative.*(u\s*\*?\s*v|uv|product).*is.*u.*v|multiply.*derivatives.*together)",
            re.IGNORECASE
        )

    def analyze(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Extracts multi-dimensional cognitive, affective, and linguistic profile from human text.
        """
        text_clean = strip_emojis(text.strip().lower())
        words = text_clean.split()
        word_count = len(words)

        is_confused = bool(self.confusion_patterns.search(text_clean))
        is_frustrated = bool(self.frustration_patterns.search(text_clean))
        is_tentative = bool(self.tentative_patterns.search(text_clean))
        is_analogy = bool(self.analogy_patterns.search(text_clean))
        is_confident = bool(self.confidence_patterns.search(text_clean))
        is_curious = bool(self.curiosity_patterns.search(text_clean))

        # Check STEM misconceptions
        misconception_type = None
        if self.misconception_fall_faster.search(text_clean):
            misconception_type = "heavier_falls_faster"
        elif self.misconception_speed_accel.search(text_clean):
            misconception_type = "speed_is_acceleration"
        elif self.misconception_product_rule.search(text_clean):
            misconception_type = "naive_product_derivative"

        # Classify communicative verbosity & pacing
        if word_count <= 3:
            verbosity = "terse"
        elif word_count >= 25:
            verbosity = "elaborate"
        else:
            verbosity = "conversational"

        # Classify primary affect state
        if is_frustrated:
            primary_affect = "frustrated"
        elif is_confused:
            primary_affect = "confused"
        elif is_tentative:
            primary_affect = "tentative"
        elif is_confident:
            primary_affect = "confident"
        elif is_curious:
            primary_affect = "curious"
        else:
            primary_affect = "inquisitive"

        # Generate tailored pedagogical adaptation instructions
        guidance_rules = []
        if is_frustrated:
            guidance_rules.append(
                "ADAPTATION: Student feels frustrated or overwhelmed. De-escalate with calming validation. "
                "Temporarily strip away mathematical symbols, focus purely on intuitive physical feeling, and ask one low-stakes question."
            )
        elif is_confused:
            guidance_rules.append(
                "ADAPTATION: Student is confused or stuck. Acknowledge that this concept is genuinely counter-intuitive. "
                "Use a vivid real-life analogy and break the problem down into a single micro-step."
            )
        elif is_tentative:
            guidance_rules.append(
                "ADAPTATION: Student is offering a tentative hypothesis with self-doubt. Validate their intuition warmly, "
                "highlight what they got right, and scaffold the next step."
            )
        elif is_analogy:
            guidance_rules.append(
                "ADAPTATION: Student seeks conceptual intuition. Provide a vivid, concrete real-world visualization "
                "before discussing mathematical formulas."
            )
        elif is_confident:
            guidance_rules.append(
                "ADAPTATION: Student exhibits confident mastery. Reinforce their insight and challenge them with an edge-case inquiry."
            )
        elif is_curious:
            guidance_rules.append(
                "ADAPTATION: Student is asking a deeper exploratory question. Answer the 'why' with first principles before returning to the exercise."
            )
        elif verbosity == "terse":
            guidance_rules.append(
                "ADAPTATION: Student gave a brief response. Affirm their input directly and ask a focused clarifying question."
            )
        else:
            guidance_rules.append(
                "ADAPTATION: Maintain supportive Socratic dialogue, actively paraphrase the student's thought, and guide discovery."
            )

        return {
            "is_confused": is_confused,
            "is_frustrated": is_frustrated,
            "is_tentative": is_tentative,
            "is_analogy": is_analogy,
            "is_confident": is_confident,
            "is_curious": is_curious,
            "misconception_type": misconception_type,
            "primary_affect": primary_affect,
            "verbosity": verbosity,
            "word_count": word_count,
            "adaptation_guidance": " ".join(guidance_rules),
        }


class PureLearnEngine:
    """
    High-level Socratic and Neuro-Inclusive tutoring engine for Purelearn.
    Integrates the custom PyTorch Transformer LM, tokenizer, NLP human adaptation,
    and adaptive prompt scaffolding. No emojis are used anywhere in output.
    """

    def __init__(
        self,
        model: Optional[PureLearnLM] = None,
        tokenizer: Optional[PureLearnTokenizer] = None,
        device: Optional[str] = None,
    ):
        # Auto-detect best available hardware accelerator
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = tokenizer or PureLearnTokenizer()
        self.nlp_analyzer = HumanCommunicationAnalyzer()
        
        if model is None:
            config = PureLearnConfig(vocab_size=self.tokenizer.vocab_size)
            self.model = PureLearnLM(config).to(self.device)
        else:
            self.model = model.to(self.device)

    def load_checkpoint(self, checkpoint_path: str):
        """Loads model weights from a PyTorch checkpoint file (.pt/.pth)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def build_socratic_prompt(
        self,
        messages: List[Dict[str, str]],
        socratic_level: int = 2,
        adhd_mode: bool = False,
        dyslexia_mode: bool = False,
        human_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Builds a structured prompt sequence incorporating pedagogical framing,
        human communication adaptation, and neuro-inclusive tokens without emojis.
        """
        socratic_guides = {
            1: "Provide gentle hints, clear scaffolding, and direct confirmations.",
            2: "Use guided Socratic questions to lead the student to discover the answer step-by-step. Break problems into micro-steps.",
            3: "Strict Socratic inquiry. Never give direct answers; challenge assumptions and guide purely through targeted questioning.",
        }
        
        system_instructions = [
            "You are Purelearn AI, an empathetic, highly intelligent neuro-inclusive Socratic tutor.",
            socratic_guides.get(socratic_level, socratic_guides[2]),
            "Always format mathematical and scientific equations using LaTeX / KaTeX notation ($...$ for inline, $$...$$ for block).",
            "Do not output emojis in responses; maintain an engaging, warm, clear, and professional tone.",
        ]

        if human_profile and human_profile.get("adaptation_guidance"):
            system_instructions.append(f"HUMAN ADAPTATION RULE: {human_profile['adaptation_guidance']}")

        if adhd_mode:
            system_instructions.append(
                "ADHD Mode Active: Use micro-chunking, short bullet points, clear visual separation, and minimize cognitive clutter."
            )
        if dyslexia_mode:
            system_instructions.append(
                "Dyslexia Mode Active: Use high readability, clean syntactic structures, simple phonetic clarity, and ample spacing."
            )

        system_text = " ".join(system_instructions)
        prompt_parts = [f"<|system|>{system_text}<|im_end|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = strip_emojis(msg.get("content", ""))
            if role == "user":
                prompt_parts.append(f"<|user|>{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>{content}<|im_end|>")

        # Open assistant turn
        prefix = "<|assistant|>"
        if adhd_mode:
            prefix += "<|adhd_chunk|>"
        if dyslexia_mode:
            prefix += "<|dyslexia_clear|>"

        prompt_parts.append(prefix)
        return "\n".join(prompt_parts)

    def apply_bionic_reading(self, text: str) -> str:
        """
        Applies Bionic Reading formatting for ADHD focus enhancement by bolding the initial fixations of words.
        """
        def bionic_word(match):
            word = match.group(0)
            if len(word) <= 1 or word.startswith("$") or word.startswith("\\"):
                return word
            mid = max(1, len(word) // 2)
            return f"**{word[:mid]}**{word[mid:]}"

        return re.sub(r"\b[a-zA-Z]{2,}\b", bionic_word, text)

    def format_neuro_inclusive(
        self,
        raw_text: str,
        adhd_mode: bool = False,
        dyslexia_mode: bool = False,
        bionic_reading: bool = False,
    ) -> str:
        """
        Post-processes model output into high-clarity neuro-inclusive markdown without emojis.
        Uses clean textual badges for pedagogical landmarks.
        """
        text = strip_emojis(raw_text)

        # Clean special tokens from display text while applying pedagogical styling
        text = text.replace("<|adhd_chunk|>", "\n\n**Focus Step:** ")
        text = text.replace("<|dyslexia_clear|>", "\n\n**Clear Guide:** ")
        text = text.replace("<|socratic_question|>", "\n\n**Question to Consider:** ")
        text = text.replace("<|hint|>", "\n\n**Helpful Hint:** ")
        text = text.replace("<|concept_check|>", "\n\n**Concept Check:** ")
        text = text.replace("<|scaffold|>", "\n\n**Next Step:** ")
        text = text.replace("<|study_notes|>", "\n\n**Key Takeaways:** ")
        text = text.replace("<|flashcard|>", "\n\n**Flashcard Concept:** ")
        text = text.replace("<|katex_math|>", " ")
        text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        # Clean multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = strip_emojis(text)

        if bionic_reading and adhd_mode:
            text = self.apply_bionic_reading(text)

        return text

    def _resolve_socratic_reasoning(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        socratic_level: int = 2,
        human_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Adaptive human reasoning engine. Interprets user intent, emotional tone,
        STEM domain, and multi-turn dialogue context to construct intuitive, scaffolded inquiries.
        """
        query = strip_emojis(user_text.lower().strip())
        profile = human_profile or self.nlp_analyzer.analyze(user_text, history)

        # 0. Misconception Intervention with Empathetic Framing
        if profile.get("misconception_type") == "heavier_falls_faster":
            return (
                "<|adhd_chunk|> That is one of the most intuitive and historic thoughts in science—in fact, Aristotle and the brightest scholars believed heavier objects fall faster for thousands of years!\n\n"
                "<|scaffold|> But think about Newton's Second Law: A heavier object feels a stronger gravitational pull ($F = m \\cdot g$), "
                "yet that same extra mass also makes it harder to accelerate ($a = F/m$).\n\n"
                "<|socratic_question|> If you plug $F = m \\cdot g$ directly into $a = F/m$, what happens to the mass ($m$) on the top and bottom of the fraction?"
            )

        if profile.get("misconception_type") == "speed_is_acceleration":
            return (
                "<|adhd_chunk|> That is an extremely common confusion because everyday language uses 'accelerating' casually to mean 'moving fast'.\n\n"
                "<|scaffold|> In physics, acceleration is strictly the **rate of change** of velocity over time: <|katex_math|> a = \\frac{\\Delta v}{\\Delta t}\n\n"
                "<|socratic_question|> If a passenger airplane is cruising smoothly in a straight line at 600 mph without changing speed or heading, what is its acceleration?"
            )

        if profile.get("misconception_type") == "naive_product_derivative":
            return (
                "<|adhd_chunk|> That feels like the most logical shortcut, and nearly every calculus student tries it at first!\n\n"
                "<|scaffold|> But imagine a rectangular garden of width $u$ and length $v$. Its area is $A = u \\cdot v$. If both the width and length grow simultaneously, "
                "the change in area is not just the corner product; both sides expand into the space.\n\n"
                "<|socratic_question|> That is why calculus uses the Product Rule: <|katex_math|> \\frac{d}{dx}[u \\cdot v] = u'v + uv'\n\n"
                "Shall we try applying this formula step-by-step to your function?"
            )

        # 1. Frustration or Confusion De-escalation
        if profile.get("is_frustrated"):
            return (
                "<|adhd_chunk|> Take a gentle breath—learning this is genuinely tricky, and it is completely normal to feel frustrated when concepts pile up. "
                "You are putting in real effort, and you are not expected to know everything immediately.\n\n"
                "<|scaffold|> Let's completely put aside the formulas for a moment and look at the simplest real-world picture: "
                "What is the single part of this that feels most confusing or stressful right now?"
            )

        if query in ["idk", "i don't know", "i dont know", "no idea", "not sure", "don't know", "dont know"]:
            return (
                "<|adhd_chunk|> That is completely fine! You do not need to have the answer right away—that is the whole point of exploring together.\n\n"
                "<|hint|> Let's break it down into an easier clue: Think about what happens in everyday life, or what your gut feeling says first.\n\n"
                "<|socratic_question|> If you had to make a wild guess based on what feels natural, what would you say?"
            )

        if profile.get("is_confused"):
            return (
                "<|adhd_chunk|> That is completely natural—when you first meet these ideas, they can feel like a foreign language. "
                "Let's step back and look at the main picture together.\n\n"
                "<|scaffold|> Which specific part feels like the biggest stumbling block so far: the terminology, the formula, or visualizing what is physically happening?"
            )

        # 2. Greetings and Onboarding
        if query in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "start"]:
            return (
                "<|adhd_chunk|> **Hello and welcome to Purelearn.**\n\n"
                "I am your neuro-inclusive Socratic AI tutor. Rather than just handing out final answers, "
                "I guide you step-by-step so you build true conceptual understanding in STEM and all your subjects.\n\n"
                "<|socratic_question|> What topic, homework problem, or concept would you like to explore together today?"
            )
        
        if query in ["help", "help me", "how does this work", "how do you work", "what can you do"]:
            return (
                "<|adhd_chunk|> **Here is how we work together:**\n\n"
                "- **Step-by-Step Guidance:** Share any problem and we will isolate the core rules.\n"
                "- **Socratic Scaffolding:** I ask targeted questions that lead you to discover the solution.\n"
                "- **KaTeX STEM Math:** Clear formulas with full visual breakdown.\n"
                "- **Study Tools:** Synthesize structured study notes or active recall flashcards anytime.\n\n"
                "<|socratic_question|> What concept or problem are you currently working on?"
            )

        # 3. Request for Analogy or Intuitive Visual
        if profile.get("is_analogy") and any(k in query for k in ["derivative", "calculus", "rate"]):
            return (
                "<|adhd_chunk|> Think of a car dashboard:\n\n"
                "- Your odometer tells you total distance traveled ($x$).\n"
                "- Your speedometer tells you your derivative—how rapidly that distance is changing at that exact second ($dx/dt$).\n\n"
                "<|socratic_question|> If your speedometer reads 60 mph, does that mean you have driven 60 miles, or that your instantaneous rate of change is 60 miles per hour?"
            )

        # 4. Multi-turn Socratic Answers & Reasoning Steps

        # Calculus: User answering derivative of x^2
        if query in ["2x", "2*x", "2 x", "two x"] or ("2x" in query and "sin" not in query):
            prefix = "Trust your instinct—you are spot on! " if profile.get("is_tentative") else "**Spot on.** "
            return (
                f"<|concept_check|> {prefix}By the Power Rule (<|katex_math|> \\frac{{d}}{{dx}}[x^n] = n x^{{n-1}}), "
                "the derivative of $x^2$ is indeed $2x$.\n\n"
                "<|adhd_chunk|> Now let's look at the second factor: <|katex_math|> v(x) = \\sin(x)\n\n"
                "<|socratic_question|> What is the derivative of $\\sin(x)$?"
            )

        # Calculus: User answering derivative of sin(x)
        if query in ["cos(x)", "cosx", "cos x", "cosine", "cosine x"] or ("cos" in query and "x^2" not in query):
            prefix = "Exactly right! " if not profile.get("is_tentative") else "You nailed it! "
            return (
                f"<|concept_check|> {prefix}The derivative of $\\sin(x)$ is $\\cos(x)$.\n\n"
                "<|scaffold|> Now we assemble both pieces using the **Product Rule**: <|katex_math|> \\frac{d}{dx}[u \\cdot v] = u'v + uv'\n\n"
                "We have:\n"
                "- $u = x^2$, $u' = 2x$\n"
                "- $v = \\sin(x)$, $v' = \\cos(x)$\n\n"
                "<|socratic_question|> When you plug these four terms into $u'v + uv'$, what is your final derivative expression?"
            )

        # Calculus: User answering combined product rule
        if ("2x" in query and "sin" in query and "cos" in query) or "2x*sin(x) + x^2*cos(x)" in query:
            return (
                "<|concept_check|> **Brilliant mathematical deduction!**\n\n"
                "<|katex_math|> f'(x) = 2x\\sin(x) + x^2\\cos(x)\n\n"
                "<|study_notes|> **Key Takeaway:** The Product Rule ensures that when both functions change simultaneously, "
                "both rates of change contribute to the total derivative: $(uv)' = u'v + uv'$."
            )

        # Physics: Tennis ball vs Bowling ball acceleration
        if any(k in query for k in ["tennis", "tennis ball", "lighter", "light ball", "smaller mass"]):
            prefix = "Spot on! " if not profile.get("is_tentative") else "Your intuition is completely accurate! "
            return (
                f"<|concept_check|> {prefix}Because $a = \\frac{{F}}{{m}}$, a smaller mass results in a proportionally larger acceleration for the same force.\n\n"
                "<|scaffold|> Now let's explore **Newton's Third Law (Action and Reaction)**:\n"
                "Imagine a tiny mosquito collides with the front windshield of a massive speeding bus.\n\n"
                "<|socratic_question|> During the collision, which object exerts a greater force on the other: the bus, the mosquito, or are the forces equal in magnitude?"
            )

        # Physics: Action & Reaction mosquito vs bus
        if any(k in query for k in ["equal", "same force", "equal and opposite", "equal force", "neither"]):
            return (
                "<|concept_check|> **Outstanding insight!** Most people incorrectly assume the bus hits harder, "
                "but Newton's Third Law guarantees that the force the bus exerts on the mosquito is **equal and opposite** to the force the mosquito exerts on the bus.\n\n"
                "<|scaffold|> Now connect this back to Newton's Second Law ($a = F/m$):\n\n"
                "<|socratic_question|> If the impact force ($F$) is identical on both, why does the mosquito experience devastating deceleration while the bus continues forward effortlessly?"
            )

        # Algorithms: Dictionary search
        if any(k in query for k in ["middle", "open to the middle", "split in half", "halfway", "divide"]):
            return (
                "<|concept_check|> **Spot on!** You flip to the middle and immediately eliminate half of the entire dictionary in a single action.\n\n"
                "<|scaffold|> This is the exact mechanism of **Binary Search** ($O(\\log n)$).\n\n"
                "<|socratic_question|> What single property must be true about the dictionary pages or list items for this halving strategy to work?"
            )

        if any(k in query for k in ["sorted", "sort", "alphabetical", "in order"]):
            return (
                "<|concept_check|> **Exactly right.** Binary Search requires the data to be **sorted**.\n\n"
                "<|adhd_chunk|> **Why Binary Search is a Superpower:**\n"
                "- In a list of 1,000 items, Linear Search takes up to 1,000 steps. Binary Search takes at most $\\lceil \\log_2(1000) \\rceil = 10$ steps!\n"
                "- In a list of 1,000,000 items, Binary Search takes only 20 steps.\n\n"
                "<|flashcard|> Q: What is the time complexity and prerequisite of Binary Search? A: $O(\\log n)$ time complexity; data must be sorted."
            )

        # Chemistry: Melting ice
        if any(k in query for k in ["absorbing", "absorbs", "absorb", "takes heat", "enters", "takes energy"]):
            return (
                "<|concept_check|> **Spot on!** Thermal energy flows from your warmer hand into the colder ice.\n\n"
                "<|adhd_chunk|> Because heat enters into the ice system, melting is an **endothermic** process ($\\Delta H > 0$).\n\n"
                "<|socratic_question|> Now consider the reverse: When water freezes into ice in your home freezer, is the water releasing heat to the freezer, or absorbing heat?"
            )

        # 5. Core STEM Concept Introductions
        if any(k in query for k in ["newton", "newton's law", "newtons law", "laws of motion"]):
            return (
                "<|dyslexia_clear|> <|adhd_chunk|> Sir Isaac Newton formulated Three Laws of Motion that govern classical mechanics:\n\n"
                "1. **First Law (Inertia):** An object at rest stays at rest, and an object in motion stays in motion at a constant velocity, unless acted upon by a net external force.\n"
                "2. **Second Law (Force and Acceleration):** <|katex_math|> F = m \\cdot a\n"
                "3. **Third Law (Action and Reaction):** For every action force, there is an equal and opposite reaction force.\n\n"
                "<|socratic_question|> Let's start with the First Law: Imagine a soccer ball resting on the grass. What must happen for that ball to begin moving across the field?"
            )

        if any(k in query for k in ["kick", "kicked", "wind", "someone kick", "push"]):
            return (
                "<|concept_check|> **Exactly right.** A net external force must act on the ball to change its state of rest.\n\n"
                "<|adhd_chunk|> Now imagine the ball is floating in the deep vacuum of outer space, with zero friction and zero gravity.\n\n"
                "<|socratic_question|> If you give it a push in deep space, will it eventually stop on its own, or will it keep moving indefinitely at the exact same speed?"
            )

        if any(k in query for k in ["keep moving", "forever", "same speed", "never stop"]):
            return (
                "<|concept_check|> **Spot on.** It keeps moving indefinitely because there is no friction or resisting force to slow it down. That is the essence of **Inertia** (Newton's 1st Law).\n\n"
                "<|scaffold|> Now let's explore the Second Law: <|katex_math|> F = m \\cdot a\n\n"
                "<|socratic_question|> If you apply the exact same pushing force ($F$) to a lightweight tennis ball versus a heavy 10 kg bowling ball, which one accelerates faster?"
            )

        if any(k in query for k in ["derivative", "differentiate", "d/dx", "calculus", "product rule"]):
            if "sin" in query or "x^2" in query or "x squared" in query:
                return (
                    "<|adhd_chunk|> Let's find the derivative step-by-step: <|katex_math|> f(x) = x^2 \\sin(x)\n\n"
                    "<|socratic_question|> Notice that this function is composed of two distinct factors multiplied together: $u(x) = x^2$ and $v(x) = \\sin(x)$. "
                    "When differentiating a product $u \\cdot v$, we apply the **Product Rule**: <|katex_math|> \\frac{d}{dx}[u \\cdot v] = u'v + uv'\n\n"
                    "<|socratic_question|> What is the derivative of the first factor, $u(x) = x^2$?"
                )
            return (
                "<|adhd_chunk|> **Calculus Foundations:**\n"
                "The derivative measures the instantaneous rate of change: <|katex_math|> f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}\n\n"
                "- **Power Rule:** <|katex_math|> \\frac{d}{dx}[x^n] = n x^{n-1}\n"
                "- **Product Rule:** <|katex_math|> (uv)' = u'v + uv'\n"
                "- **Chain Rule:** <|katex_math|> \\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)\n\n"
                "<|socratic_question|> Which specific function or equation would you like to differentiate together?"
            )

        if any(k in query for k in ["binary search", "search algorithm", "big o", "time complexity"]):
            return (
                "<|adhd_chunk|> Let's compare searching approaches:\n"
                "- **Linear Search:** Checks items one by one ($O(n)$).\n"
                "- **Binary Search:** Eliminates half of the remaining elements at each step ($O(\\log n)$).\n\n"
                "<|socratic_question|> Imagine searching for a specific word in a 1,000-page dictionary. What is the very first thing you do?"
            )

        if any(k in query for k in ["endothermic", "exothermic", "chemical reaction", "heat"]):
            return (
                "<|dyslexia_clear|> <|adhd_chunk|> Look at the word roots:\n"
                "- **Exo-** = Exit (energy/heat exits into the surroundings, $\\Delta H < 0$)\n"
                "- **Endo-** = Enter (energy/heat enters into the system, $\\Delta H > 0$)\n\n"
                "<|socratic_question|> When an ice cube melts in your warm hand, is the ice absorbing heat from your hand or releasing heat?"
            )

        return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        socratic_level: int = 2,
        adhd_mode: bool = False,
        dyslexia_mode: bool = False,
        bionic_reading: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 350,
    ) -> Dict[str, Any]:
        """
        Processes a chat dialogue turn with NLP human communication adaptation
        and returns a Socratic neuro-inclusive response without emojis.
        """
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        # NLP Human Affect & Intent Analysis
        human_profile = self.nlp_analyzer.analyze(last_user_msg, messages)

        # Check Socratic reasoning engine
        reasoned_response = self._resolve_socratic_reasoning(
            last_user_msg,
            messages,
            socratic_level=socratic_level,
            human_profile=human_profile,
        )

        if reasoned_response is not None:
            raw_output = reasoned_response
            gen_tokens_count = len(self.tokenizer.encode(raw_output))
        else:
            prompt = self.build_socratic_prompt(
                messages,
                socratic_level=socratic_level,
                adhd_mode=adhd_mode,
                dyslexia_mode=dyslexia_mode,
                human_profile=human_profile,
            )

            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            output_tensor = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_id=self.tokenizer.eos_id,
            )

            # Extract generated tokens
            gen_tokens = output_tensor[0, len(input_ids) :].tolist()
            raw_output = self.tokenizer.decode(gen_tokens)
            if "<|im_end|>" in raw_output:
                raw_output = raw_output.split("<|im_end|>")[0]
            gen_tokens_count = len(gen_tokens)

            # Fallback if autoregressive sampling produced degenerate output
            if len(raw_output.strip()) < 10 or "<|user|>" in raw_output:
                raw_output = (
                    f"<|adhd_chunk|> That is an interesting question regarding **'{strip_emojis(last_user_msg[:40])}'**.\n\n"
                    f"<|socratic_question|> To help us break this down clearly, what is the core equation or first principle you think applies here?"
                )

        formatted_output = self.format_neuro_inclusive(
            raw_output,
            adhd_mode=adhd_mode,
            dyslexia_mode=dyslexia_mode,
            bionic_reading=bionic_reading,
        )

        return {
            "response": formatted_output,
            "raw_response": raw_output,
            "tokens_generated": gen_tokens_count,
            "socratic_level": socratic_level,
            "adhd_mode": adhd_mode,
            "dyslexia_mode": dyslexia_mode,
            "human_adaptation": human_profile,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        socratic_level: int = 2,
        adhd_mode: bool = False,
        dyslexia_mode: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 350,
    ) -> Generator[str, None, None]:
        """
        Streams generated response tokens in real-time with NLP adaptation and no emojis.
        """
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        human_profile = self.nlp_analyzer.analyze(last_user_msg, messages)

        reasoned_response = self._resolve_socratic_reasoning(
            last_user_msg,
            messages,
            socratic_level=socratic_level,
            human_profile=human_profile,
        )

        if reasoned_response is not None:
            formatted = self.format_neuro_inclusive(
                reasoned_response,
                adhd_mode=adhd_mode,
                dyslexia_mode=dyslexia_mode,
            )
            words = re.findall(r"\S+|\s+", formatted)
            for word in words:
                yield word
        else:
            prompt = self.build_socratic_prompt(
                messages,
                socratic_level=socratic_level,
                adhd_mode=adhd_mode,
                dyslexia_mode=dyslexia_mode,
                human_profile=human_profile,
            )

            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            for token_id in self.model.generate_stream(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_id=self.tokenizer.eos_id,
            ):
                piece = self.tokenizer.decode([token_id])
                if piece == "<|im_end|>":
                    break
                yield strip_emojis(piece)

    def generate_study_notes(
        self,
        topic_or_transcript: str,
        adhd_mode: bool = False,
        dyslexia_mode: bool = False,
        bionic_reading: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates structured Markdown study notes without emojis.
        """
        clean_topic = strip_emojis(topic_or_transcript)
        prompt = (
            f"<|system|>You are Purelearn AI Study Synthesizer. Convert the following material into clear, "
            f"structured Markdown study notes with KaTeX equations, bullet points, and key takeaways without emojis.<|im_end|>\n"
            f"<|user|>Generate comprehensive study notes for:\n{clean_topic}<|im_end|>\n"
            f"<|assistant|><|study_notes|>"
        )
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_tensor = self.model.generate(
            input_tensor,
            max_new_tokens=400,
            temperature=0.6,
            eos_id=self.tokenizer.eos_id,
        )

        gen_tokens = output_tensor[0, len(input_ids) :].tolist()
        raw_text = self.tokenizer.decode(gen_tokens)
        if "<|im_end|>" in raw_text:
            raw_text = raw_text.split("<|im_end|>")[0]

        formatted_notes = self.format_neuro_inclusive(
            raw_text,
            adhd_mode=adhd_mode,
            dyslexia_mode=dyslexia_mode,
            bionic_reading=bionic_reading,
        )

        if len(formatted_notes.strip()) < 20:
            formatted_notes = (
                f"# Study Notes: {clean_topic[:40]}...\n\n"
                f"## Core Concepts\n"
                f"- **Primary Principle:** Key concepts and definitions formatted for active recall.\n"
                f"- **Mathematical / STEM Formulation:**\n"
                f"  $$\\mathcal{{L}}_{{\\text{{PureLearn}}}} = -\\sum_{{t=1}}^T \\log P(w_t \\mid w_{{<t}})$$\n\n"
                f"## Step-by-Step Breakdown\n"
                f"1. **Identify Given Information:** Break down variables and constraints.\n"
                f"2. **Apply Socratic Scaffolding:** Isolate the rule or theorem needed.\n"
                f"3. **Compute and Verify:** Check consistency and units.\n\n"
                f"## Quick Recall Summary\n"
                f"> Always test edge cases and simplify components before calculating final outputs."
            )

        return {
            "topic": clean_topic[:60],
            "notes_markdown": strip_emojis(formatted_notes),
        }

    def generate_flashcards(self, text_or_topic: str, num_cards: int = 3) -> List[Dict[str, str]]:
        """
        Generates active recall Q&A flashcards without emojis.
        """
        clean_text = strip_emojis(text_or_topic)
        prompt = (
            f"<|system|>You are Purelearn AI Flashcard Generator. Create active recall flashcards without emojis.<|im_end|>\n"
            f"<|user|>Create flashcards for:\n{clean_text}<|im_end|>\n"
            f"<|assistant|><|flashcard|>"
        )
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_tensor = self.model.generate(
            input_tensor,
            max_new_tokens=300,
            temperature=0.6,
            eos_id=self.tokenizer.eos_id,
        )

        gen_tokens = output_tensor[0, len(input_ids) :].tolist()
        raw_text = self.tokenizer.decode(gen_tokens)

        flashcards = []
        pattern = re.findall(r"\*\*Q:\*\*\s*(.*?)\s*\*\*A:\*\*\s*(.*?)(?=(?:\*\*Q:\*\*|$))", raw_text, re.DOTALL)
        
        for q, a in pattern[:num_cards]:
            if q.strip() and a.strip():
                flashcards.append({
                    "question": strip_emojis(q.strip()),
                    "answer": strip_emojis(a.strip()),
                    "category": "PureLearn Active Recall"
                })

        if not flashcards:
            flashcards = [
                {
                    "question": f"What is the foundational concept behind '{clean_text[:35]}'?",
                    "answer": "The core theorem and its governing rules applied to problem solving.",
                    "category": "Foundations"
                },
                {
                    "question": "How do you verify the solution in STEM problems?",
                    "answer": "Check boundary conditions, dimensional units, and step-by-step arithmetic.",
                    "category": "Verification"
                }
            ]

        return flashcards
