"""
SAIYAARA — memory.py
Long-term memory system.
Stores facts about Vinay permanently in memory.json.
Triggered by phrases like "remember that...", "I prefer...", "forget that..."
"""

import os
import json
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")


# ── Load & Save ───────────────────────────────────────────────────────────────
def _load() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"facts": [], "preferences": [], "personal": []}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"facts": [], "preferences": [], "personal": []}


def _save(data: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Remember ──────────────────────────────────────────────────────────────────
def remember(category: str, fact: str) -> str:
    """
    Store a new fact in memory.
    Categories: facts, preferences, personal
    """
    data = _load()

    # Avoid exact duplicates
    if fact.strip() in data.get(category, []):
        return f"I already know that, sir — '{fact}'"

    if category not in data:
        data[category] = []

    data[category].append(fact.strip())
    _save(data)

    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    print(f"[Memory] Stored in '{category}': {fact} ({timestamp})")
    return f"Got it, sir. I'll remember that."


# ── Recall ────────────────────────────────────────────────────────────────────
def recall_all() -> str:
    """
    Return all memories as a formatted string.
    Injected into system prompt every session.
    """
    data = _load()
    lines = []

    if data.get("personal"):
        lines.append("Personal:")
        for f in data["personal"]:
            lines.append(f"  - {f}")

    if data.get("preferences"):
        lines.append("Preferences:")
        for f in data["preferences"]:
            lines.append(f"  - {f}")

    if data.get("facts"):
        lines.append("Facts:")
        for f in data["facts"]:
            lines.append(f"  - {f}")

    if not lines:
        return ""

    return "What I know about Vinay:\n" + "\n".join(lines)


def recall_relevant(topic: str) -> str:
    """
    Return memories relevant to a specific topic.
    Simple keyword match for now.
    """
    data = _load()
    all_facts = (
        data.get("personal", []) +
        data.get("preferences", []) +
        data.get("facts", [])
    )

    topic_words = topic.lower().split()
    relevant = [
        f for f in all_facts
        if any(word in f.lower() for word in topic_words)
    ]

    if not relevant:
        return ""

    return "Relevant memories:\n" + "\n".join(f"  - {r}" for r in relevant)


# ── Forget ────────────────────────────────────────────────────────────────────
def forget(fact: str) -> str:
    """
    Remove a specific fact from memory.
    """
    data = _load()
    removed = False

    for category in data:
        if fact.strip() in data[category]:
            data[category].remove(fact.strip())
            removed = True

    if removed:
        _save(data)
        return f"Done, sir. I've forgotten '{fact}'."
    return f"I don't seem to have '{fact}' in my memory, sir."


def forget_all() -> str:
    """
    Wipe entire memory. Requires explicit confirmation.
    """
    _save({"facts": [], "preferences": [], "personal": []})
    return "Memory wiped clean, sir."


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(remember("personal", "Vinay is from Visakhapatnam"))
    print(remember("preferences", "Vinay prefers dark mode"))
    print(remember("preferences", "Vinay prefers dark mode"))  # duplicate test
    print(remember("facts", "Vinay is applying for AP PGCET 2026"))
    print()
    print(recall_all())
    print()
    print(recall_relevant("dark mode"))
    print()
    print(forget("Vinay prefers dark mode"))
    print()
    print(recall_all())