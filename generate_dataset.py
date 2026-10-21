"""
High-Fidelity Socratic & Human Communication Dataset Generator for Purelearn.
Generates thousands of diverse, natural human-to-AI dialogues covering STEM concepts,
conversational nuances, cognitive adaptations, misconceptions, and Socratic pedagogy.
Completely free of emojis.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from purelearn_model.dataset import strip_emojis


TOPICS_DATA = [
    # Mathematics
    {
        "category": "Mathematics - Calculus",
        "concept": "Product Rule",
        "formula": r"\frac{d}{dx}[u \cdot v] = u'v + uv'",
        "scenario": r"f(x) = x^3 \cdot \cos(x)",
        "misconception": "Just multiplying individual derivatives: f'(x) = 3x^2 * (-sin(x))",
        "real_world": "Finding the rate at which area changes when both width and length change over time",
        "parts": ("u(x) = x^3", "v(x) = \\cos(x)"),
        "derivatives": ("u'(x) = 3x^2", "v'(x) = -\\sin(x)"),
    },
    {
        "category": "Mathematics - Calculus",
        "concept": "Chain Rule",
        "formula": r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)",
        "scenario": r"f(x) = \sin(x^2 + 3x)",
        "misconception": "Only differentiating the outside function and forgetting the inside derivative",
        "real_world": "Rate of inflation affecting purchasing power which affects family expenditure",
        "parts": ("outer: \\sin(u)", "inner: u = x^2 + 3x"),
        "derivatives": ("outer derivative: \\cos(u)", "inner derivative: 2x + 3"),
    },
    {
        "category": "Mathematics - Calculus",
        "concept": "Fundamental Theorem of Calculus",
        "formula": r"\int_a^b f(x) \, dx = F(b) - F(a)",
        "scenario": r"\int_0^2 (3x^2 + 2) \, dx",
        "misconception": "Thinking differentiation and integration are completely unrelated operations",
        "real_world": "Computing total distance traveled by integrating an automobile's varying speedometer reading",
        "parts": ("integrand: 3x^2 + 2", "antiderivative: F(x) = x^3 + 2x"),
        "derivatives": ("derivative of F(x) yields 3x^2 + 2", "evaluated from 0 to 2"),
    },
    {
        "category": "Mathematics - Linear Algebra",
        "concept": "Eigenvalues and Eigenvectors",
        "formula": r"A \vec{v} = \lambda \vec{v}",
        "scenario": "A 2x2 shearing matrix acting on a 2D plane",
        "misconception": "Assuming all vectors change direction when multiplied by a matrix",
        "real_world": "Google PageRank algorithm calculating node authority via the principal eigenvector",
        "parts": ("matrix: A", "eigenvector: \\vec{v}", "scalar eigenvalue: \\lambda"),
        "derivatives": ("characteristic equation: \\det(A - \\lambda I) = 0", "scaling without rotation"),
    },
    # Physics
    {
        "category": "Physics - Mechanics",
        "concept": "Newton's First Law (Inertia)",
        "formula": r"\sum \vec{F} = 0 \implies \vec{v} = \text{const}",
        "scenario": "A puck gliding across frictionless ice in a hockey rink",
        "misconception": "Believing an object requires constant force just to keep moving at a steady velocity",
        "real_world": "A spacecraft continuing on its trajectory through deep space with engines turned off",
        "parts": ("net external force: 0", "velocity: constant vector"),
        "derivatives": ("acceleration: 0", "mass resists changes in state of motion"),
    },
    {
        "category": "Physics - Mechanics",
        "concept": "Newton's Second Law",
        "formula": r"\vec{F}_{\text{net}} = m \cdot \vec{a}",
        "scenario": "A heavy transport truck braking compared to a lightweight compact car",
        "misconception": "Conflating force with momentum or assuming heavier objects naturally accelerate faster",
        "real_world": "Rocket thrusters creating acceleration as fuel mass diminishes during flight",
        "parts": ("force: F", "mass: m", "acceleration: a"),
        "derivatives": ("doubling mass halves acceleration for fixed force", "a = F / m"),
    },
    {
        "category": "Physics - Mechanics",
        "concept": "Conservation of Momentum",
        "formula": r"m_1 u_1 + m_2 u_2 = m_1 v_1 + m_2 v_2",
        "scenario": "Two billiard balls colliding head-on on a smooth table",
        "misconception": "Thinking momentum is lost in an inelastic collision when only kinetic energy converts to heat",
        "real_world": "Recoil felt by a rifle when firing a high-velocity bullet",
        "parts": ("initial momentum: p_i", "final momentum: p_f"),
        "derivatives": ("total momentum before equals total after in isolated systems", "closed system condition"),
    },
    {
        "category": "Physics - Thermodynamics",
        "concept": "First Law of Thermodynamics",
        "formula": r"\Delta U = Q - W",
        "scenario": "A gas cylinder compressed by a piston while heat is added",
        "misconception": "Treating heat and temperature as the exact same physical quantity",
        "real_world": "Internal combustion engines converting thermal energy into mechanical drive",
        "parts": ("internal energy change: \\Delta U", "heat added: Q", "work done by system: W"),
        "derivatives": ("conservation of energy", "temperature relates to average kinetic energy"),
    },
    # Computer Science
    {
        "category": "Computer Science - Algorithms",
        "concept": "Binary Search",
        "formula": r"O(\log n)",
        "scenario": "Searching for an item among 1,000,000 sorted records",
        "misconception": "Trying to apply binary search on an unsorted list without sorting first",
        "real_world": "Looking up a contact in an alphabetically indexed directory",
        "parts": ("sorted collection", "midpoint comparison", "search space halved each step"),
        "derivatives": ("1,000,000 items checked in roughly 20 comparisons", "logarithmic vs linear complexity"),
    },
    {
        "category": "Computer Science - Algorithms",
        "concept": "Breadth-First Search (BFS)",
        "formula": r"O(V + E)",
        "scenario": "Finding the shortest route between two cities on a road network",
        "misconception": "Thinking BFS uses a stack instead of a FIFO queue",
        "real_world": "Social networks calculating degrees of separation between friends",
        "parts": ("vertices: V", "edges: E", "queue data structure"),
        "derivatives": ("explores neighbor nodes layer by layer", "optimal for unweighted graphs"),
    },
    {
        "category": "Computer Science - Algorithms",
        "concept": "Dynamic Programming",
        "formula": r"T(n) = T(n-1) + T(n-2) \implies O(n)",
        "scenario": "Computing Fibonacci numbers with memoization",
        "misconception": "Confusing divide-and-conquer with overlapping subproblems",
        "real_world": "DNA sequence alignment algorithms finding maximum genetic similarity",
        "parts": ("optimal substructure", "overlapping subproblems", "memoization table"),
        "derivatives": ("transforms O(2^n) exponential recursion into O(n) linear table lookup", "caching results"),
    },
    # Chemistry
    {
        "category": "Chemistry - Chemical Equilibrium",
        "concept": "Le Chatelier's Principle",
        "formula": r"K_{\text{eq}} = \frac{[\text{Products}]^c}{[\text{Reactants}]^a}",
        "scenario": "Compressing a closed reaction vessel containing nitrogen and hydrogen gas to synthesize ammonia",
        "misconception": "Assuming equilibrium means equal concentrations rather than equal reaction rates",
        "real_world": "Industrial Haber-Bosch process optimizing fertilizer production",
        "parts": ("forward reaction rate", "reverse reaction rate", "pressure/concentration disturbance"),
        "derivatives": ("system shifts to counteract applied stress", "volume reduction favors fewer moles of gas"),
    },
    {
        "category": "Chemistry - Acid-Base",
        "concept": "pH Scale and Hydrogen Ion Concentration",
        "formula": r"\text{pH} = -\log_{10}[H^+]",
        "scenario": "Diluting pure water with hydrochloric acid",
        "misconception": "Thinking a change of 1 pH unit is a linear 10% change rather than a 10-fold exponential factor",
        "real_world": "Blood buffer systems maintaining human physiological pH between 7.35 and 7.45",
        "parts": ("hydronium ion concentration: [H^+]", "logarithmic scale"),
        "derivatives": ("pH 3 is 100 times more acidic than pH 5", "neutral at pH 7"),
    },
]


def generate_conversational_dialogue(topic: Dict[str, Any], variant_id: int) -> Dict[str, Any]:
    """
    Generates a natural, intelligent, multi-turn human conversation reflecting
    human communication traits (hesitation, real-world framing, misconceptions,
    neuro-inclusive scaffolding, and empathy) without emojis.
    """
    archetype = variant_id % 6
    concept = topic["concept"]
    formula = topic["formula"]
    scenario = topic["scenario"]
    misconception = topic["misconception"]
    real_world = topic["real_world"]
    parts = topic["parts"]
    derivatives = topic["derivatives"]

    if archetype == 0:
        # Archetype 0: Hesitant & Uncertain Learner
        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"I am really struggling to understand {concept}. It feels confusing when I look at {scenario}. Where do I even begin?"
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> That is completely normal. Many students find {concept} unintuitive at first glance.\n\n"
                    f"Let's ignore the complicated notation for a moment and look at the real-world idea:\n"
                    f"- {real_world}.\n\n"
                    f"<|socratic_question|> In that situation, what is the primary thing that changes first?"
                )
            },
            {
                "role": "user",
                "content": f"I think it's the underlying input or initial condition, like in {scenario}."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> You are thinking about this in the exact right way.\n\n"
                    f"<|scaffold|> Here is the mathematical formula that expresses that very relationship:\n"
                    f"<|katex_math|> {formula}\n\n"
                    f"<|socratic_question|> Looking at the parts of this equation, what happens if we double the first variable?"
                )
            },
            {
                "role": "user",
                "content": "It causes the result to change proportionally based on how the variables multiply together."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Exactly right. You just derived the core mechanism yourself.\n\n"
                    f"<|study_notes|> Key Takeaways for {concept}:\n"
                    f"- Governing Rule: ${formula}$\n"
                    f"- Intuitive Picture: {real_world}\n"
                    f"- Method: Isolate the primary factors before computing."
                )
            }
        ]

    elif archetype == 1:
        # Archetype 1: Common Misconception Confrontation
        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"For {concept}, why can't I just simplify things by doing this: {misconception}?"
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> That is one of the most common and logical traps learners fall into. It is a great question to ask.\n\n"
                    f"Let's test why that shortcut breaks down. Consider {scenario}.\n\n"
                    f"<|socratic_question|> If both parts are changing at the same time, does changing one piece affect the contribution of the second piece?"
                )
            },
            {
                "role": "user",
                "content": "Yes, because they interact with each other rather than acting independently."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> Spot on. That is precisely why we need the full formulation:\n"
                    f"<|katex_math|> {formula}\n\n"
                    f"<|scaffold|> Notice how the cross terms account for that interaction.\n\n"
                    f"<|socratic_question|> What would happen to the result if one of the components stayed completely constant?"
                )
            },
            {
                "role": "user",
                "content": "The derivative or change of the constant part would be zero, leaving only the remaining term."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Outstanding deduction. That verifies why the general rule works in every single case.\n\n"
                    f"<|study_notes|> Concept Check for {concept}:\n"
                    f"- Fallacy to Avoid: {misconception}\n"
                    f"- True Formulation: ${formula}$\n"
                    f"- Reason: Both factors interact simultaneously."
                )
            }
        ]

    elif archetype == 2:
        # Archetype 2: Everyday Analogy Seeking
        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"Can you give me an analogy for {concept}? I understand the formula {formula}, but I can't visualize it."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|dyslexia_clear|> <|adhd_chunk|> Absolutely. Physical intuition makes formulas stick.\n\n"
                    f"Imagine this scenario: {real_world}.\n\n"
                    f"<|socratic_question|> When you picture that happening in your mind, what is the main trade-off taking place?"
                )
            },
            {
                "role": "user",
                "content": f"As one part increases, the other has to adjust to maintain the relationship described in {scenario}."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> Exactly. Now connect that picture back to the equation:\n"
                    f"<|katex_math|> {formula}\n\n"
                    f"<|scaffold|> The symbols are just concise shorthand for the physical trade-off you visualized.\n\n"
                    f"<|socratic_question|> How does this help you predict what happens when values approach zero or infinity?"
                )
            },
            {
                "role": "user",
                "content": "It lets me test the extreme limits intuitively before doing heavy algebra."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Brilliant. Testing boundary conditions is the secret of expert physicists and mathematicians.\n\n"
                    f"<|flashcard|> Q: What is the physical intuition behind {concept}? A: {real_world}, represented by ${formula}$."
                )
            }
        ]

    elif archetype == 3:
        # Archetype 3: Step-by-Step Problem Solving & Verification
        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"I have a homework problem on {concept} involving {scenario}. How do I set it up?"
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Let's solve it step-by-step. First, we identify our given factors:\n"
                    f"- Component 1: {parts[0]}\n"
                    f"- Component 2: {parts[1]}\n\n"
                    f"<|socratic_question|> What is the specific rule or governing relationship connecting these parts?"
                )
            },
            {
                "role": "user",
                "content": f"We use {formula}."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> Correct.\n\n"
                    f"<|scaffold|> Now calculate the corresponding values or rates of change for each individual part:\n"
                    f"- {derivatives[0]}\n"
                    f"- {derivatives[1]}\n\n"
                    f"<|socratic_question|> What do you get when you combine them into the formula?"
                )
            },
            {
                "role": "user",
                "content": f"We substitute {derivatives[0]} and {derivatives[1]} directly into ${formula}$ to produce the final simplified answer."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Spot on. That is the complete solution.\n\n"
                    f"<|study_notes|> Step-by-Step Procedure for {concept}:\n"
                    f"1. Break the problem into separate factors: {parts[0]} and {parts[1]}.\n"
                    f"2. Compute individual components: {derivatives[0]}.\n"
                    f"3. Assemble via ${formula}$."
                )
            }
        ]

    elif archetype == 4:
        # Archetype 4: Overwhelmed / ADHD Micro-Chunking
        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"There are way too many symbols in {concept}. My brain is totally overloaded."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Take a deep breath. We are going to strip away 90% of the visual clutter.\n\n"
                    f"Focus only on this single piece:\n"
                    f"- First Piece: {parts[0]}\n\n"
                    f"<|socratic_question|> Looking only at that first piece and nothing else, what does it do on its own?"
                )
            },
            {
                "role": "user",
                "content": f"On its own, it just represents {derivatives[0]}."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> Exactly. That was painless, right?\n\n"
                    f"<|adhd_chunk|> Now look at only the second piece:\n"
                    f"- Second Piece: {parts[1]}\n\n"
                    f"<|socratic_question|> What is its individual behavior?"
                )
            },
            {
                "role": "user",
                "content": f"Its behavior is {derivatives[1]}."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Look at that: You just understood both pieces without any overwhelm.\n\n"
                    f"All {concept} does is add them together:\n"
                    f"<|katex_math|> {formula}\n\n"
                    f"<|study_notes|> Focus Rule: When a STEM formula looks intimidating, cover half of it and master the first term before looking at the rest."
                )
            }
        ]

    else:
        # Archetype 5: Inquisitive Student Exploring Edge Cases
        dialogue = [
            {"role": "system", "content": "You are Purelearn AI, a neuro-inclusive Socratic tutor that adapts to human communication."},
            {
                "role": "user",
                "content": f"Does {concept} still hold true in extreme conditions, or does it break down?"
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> That is the mark of a true scientist—testing the boundaries of a theory.\n\n"
                    f"The governing relationship is:\n"
                    f"<|katex_math|> {formula}\n\n"
                    f"<|socratic_question|> In {scenario}, what happens if the primary variable approaches zero or infinity?"
                )
            },
            {
                "role": "user",
                "content": "As the variable approaches extreme limits, certain terms dominate while others become negligible."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|concept_check|> Precisely. That is where asymptotic analysis and boundary conditions come into play.\n\n"
                    f"<|scaffold|> For instance, in classical physics or standard calculus, the rule holds as long as our foundational assumptions remain intact.\n\n"
                    f"<|socratic_question|> What physical constraint in the real world might violate those ideal assumptions?"
                )
            },
            {
                "role": "user",
                "content": "Things like air resistance, relativistic speeds, or discrete quantum effects."
            },
            {
                "role": "assistant",
                "content": (
                    f"<|adhd_chunk|> Brilliant insight. You have connected introductory principles with modern physical theories.\n\n"
                    f"<|study_notes|> Boundary Analysis for {concept}:\n"
                    f"- Ideal Model: ${formula}$\n"
                    f"- Boundary Constraints: Always check for friction, non-linear effects, and extreme scaling limits."
                )
            }
        ]

    # Sanitize any accidental emoji from dialogue turns
    for turn in dialogue:
        turn["content"] = strip_emojis(turn["content"])

    return {
        "id": f"purelearn_human_socratic_{variant_id:06d}",
        "category": topic["category"],
        "concept": concept,
        "dialogue": dialogue,
    }


def generate_dataset(num_samples: int = 2500, output_file: str = "data/purelearn_curriculum.jsonl"):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset Generator] Generating {num_samples:,} refined human-centric Socratic dialogues across STEM domains...")
    samples = []

    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id in range(1, num_samples + 1):
            topic = random.choice(TOPICS_DATA)
            record = generate_conversational_dialogue(topic, sample_id)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples.append(record)

    print(f"[Dataset Generator] Successfully generated {len(samples):,} refined, emoji-free training samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Refined Purelearn Socratic Training Dataset")
    parser.add_argument("--num_samples", type=int, default=2500, help="Number of dialogues to generate")
    parser.add_argument("--output_file", type=str, default="data/purelearn_curriculum.jsonl", help="Output file path")
    args = parser.parse_args()

    generate_dataset(args.num_samples, args.output_file)
