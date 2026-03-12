"""
SAIYAARA — memory.py
Long-term memory system.

Two levels of memory:
1. Facts    — things Vinay explicitly told SAIYAARA ("remember that...")
2. Understanding — patterns SAIYAARA noticed herself over time (auto-built every 10 conversations)

Also tracks:
- Conversation counter (triggers understanding builder every 10 conversations)
- Recent conversation summaries (used to build understanding)
- Curiosity cooldown (prevents asking questions too frequently)
"""

import os
import json
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & SAVE — internal helpers, not called directly from outside
# ─────────────────────────────────────────────────────────────────────────────

def _load_memory() -> dict:
    """
    Loads memory.json from disk.
    If file doesn't exist or is corrupted, returns a clean empty structure.
    """
    empty = {
        "facts": [],
        "preferences": [],
        "personal": [],
        "understanding": [],        # patterns SAIYAARA noticed herself
        "conversation_count": 0,    # how many conversations have happened total
        "recent_summaries": [],     # last 10 conversation summaries (for understanding builder)
        "last_curiosity_message": 0 # message number when curiosity last fired (cooldown tracker)
    }
    if not os.path.exists(MEMORY_FILE):
        return empty
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Make sure all keys exist even if file is from older version
            for key, default in empty.items():
                if key not in data:
                    data[key] = default
            return data
    except:
        return empty


def _save_memory(data: dict):
    """
    Saves the memory dictionary back to memory.json on disk.
    """
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# REMEMBER — store a fact Vinay explicitly told SAIYAARA
# ─────────────────────────────────────────────────────────────────────────────

def remember(category: str, fact: str) -> str:
    """
    Stores a new fact that Vinay explicitly shared.
    Categories: facts, preferences, personal

    Prevents storing exact duplicates.
    Called from brain.py when router detects a memory store request.
    """
    data = _load_memory()

    # Don't store if we already have this exact fact
    if fact.strip() in data.get(category, []):
        return f"I already know that, sir — '{fact}'"

    if category not in data:
        data[category] = []

    data[category].append(fact.strip())
    _save_memory(data)

    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    print(f"[Memory] Stored in '{category}': {fact} ({timestamp})")
    return f"Got it, sir. I'll remember that."


# ─────────────────────────────────────────────────────────────────────────────
# RECALL — retrieve memories for injection into system prompt
# ─────────────────────────────────────────────────────────────────────────────

def recall_all() -> str:
    """
    Returns ALL stored facts + understanding as a formatted string.
    Used by brain.py's memory selector to pick what's relevant.
    """
    data = _load_memory()
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

    if data.get("understanding"):
        lines.append("Things I've noticed about Vinay over time:")
        for u in data["understanding"]:
            lines.append(f"  - {u}")

    if not lines:
        return ""

    return "What I know about Vinay:\n" + "\n".join(lines)


def get_total_memory_count() -> int:
    """
    Returns total number of stored facts across all categories.
    Used to skip the Groq memory selector call when memory is nearly empty
    (saves tokens when there's nothing useful to inject anyway).
    """
    data = _load_memory()
    return (
        len(data.get("personal", [])) +
        len(data.get("preferences", [])) +
        len(data.get("facts", [])) +
        len(data.get("understanding", []))
    )


# ─────────────────────────────────────────────────────────────────────────────
# FORGET — remove a fact from memory
# ─────────────────────────────────────────────────────────────────────────────

def forget(fact: str) -> str:
    """
    Removes a specific fact from memory.
    Searches across all categories (personal, preferences, facts).
    Called from brain.py when Vinay says "forget that..."
    """
    data = _load_memory()
    removed = False

    for category in ["personal", "preferences", "facts"]:
        if fact.strip() in data.get(category, []):
            data[category].remove(fact.strip())
            removed = True

    if removed:
        _save_memory(data)
        return f"Done, sir. I've forgotten '{fact}'."
    return f"I don't seem to have '{fact}' in my memory, sir."


def forget_all() -> str:
    """
    Wipes the entire memory clean.
    Keeps the structure intact but empties all lists.
    Should only be called after explicit confirmation from Vinay.
    """
    _save_memory({
        "facts": [],
        "preferences": [],
        "personal": [],
        "understanding": [],
        "conversation_count": 0,
        "recent_summaries": [],
        "last_curiosity_message": 0
    })
    return "Memory wiped clean, sir."


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION TRACKING — counts conversations, stores summaries
# ─────────────────────────────────────────────────────────────────────────────

def record_conversation_ended(summary: str):
    """
    Called at the end of each conversation session (when server disconnects).
    Does two things:
    1. Increments the conversation counter
    2. Stores a short summary of this conversation (kept last 10 only)

    The summary is used every 10 conversations to build 'understanding'.
    """
    data = _load_memory()

    # Increment total conversation count
    data["conversation_count"] = data.get("conversation_count", 0) + 1

    # Store this conversation's summary, keep only the last 10
    summaries = data.get("recent_summaries", [])
    summaries.append({
        "date": datetime.now().strftime("%d %b %Y"),
        "summary": summary.strip()
    })
    data["recent_summaries"] = summaries[-10:]  # keep only last 10

    _save_memory(data)
    print(f"[Memory] Conversation #{data['conversation_count']} recorded.")
    return data["conversation_count"]


def get_conversation_count() -> int:
    """
    Returns total number of conversations had so far.
    Used to decide when to trigger the understanding builder (every 10).
    """
    return _load_memory().get("conversation_count", 0)


def get_recent_summaries() -> list:
    """
    Returns the last 10 conversation summaries.
    Fed into the understanding builder so Groq can find patterns.
    """
    return _load_memory().get("recent_summaries", [])


# ─────────────────────────────────────────────────────────────────────────────
# UNDERSTANDING — patterns SAIYAARA notices herself over time
# ─────────────────────────────────────────────────────────────────────────────

def save_understanding(new_insights: list):
    """
    Stores new pattern-based insights that SAIYAARA figured out herself.
    These are NOT facts Vinay told her — they're things she noticed.

    Example: "Vinay goes quiet when something's bothering him, rarely says it directly"

    Called from brain.py's understanding builder after every 10 conversations.
    Adds new insights, avoids duplicates, keeps total under 20.
    """
    data = _load_memory()
    existing = data.get("understanding", [])

    added = 0
    for insight in new_insights:
        insight = insight.strip()
        # Skip if too similar to something already stored (basic duplicate check)
        if insight and insight not in existing:
            existing.append(insight)
            added += 1

    # Keep only the most recent 20 understandings to avoid bloat
    data["understanding"] = existing[-20:]
    _save_memory(data)
    print(f"[Memory] Understanding updated — {added} new insights added.")


# ─────────────────────────────────────────────────────────────────────────────
# CURIOSITY COOLDOWN — prevents asking questions too frequently
# ─────────────────────────────────────────────────────────────────────────────

def get_last_curiosity_message() -> int:
    """
    Returns the message number when SAIYAARA last asked a curiosity question.
    Used to enforce a cooldown — don't ask again too soon.
    """
    return _load_memory().get("last_curiosity_message", 0)


def update_curiosity_fired(current_message_number: int):
    """
    Records that SAIYAARA just asked a curiosity question.
    Updates the cooldown tracker so she doesn't ask again too soon.
    """
    data = _load_memory()
    data["last_curiosity_message"] = current_message_number
    _save_memory(data)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify memory works
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(remember("personal", "Vinay is from Visakhapatnam"))
    print(remember("preferences", "Vinay prefers dark mode"))
    print(remember("preferences", "Vinay prefers dark mode"))  # duplicate test
    print(remember("facts", "Vinay is building SAIYAARA"))
    print()
    print(recall_all())
    print()
    print(f"Total memory count: {get_total_memory_count()}")
    print()
    print(forget("Vinay prefers dark mode"))
    print()
    print(recall_all())