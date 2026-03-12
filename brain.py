"""
SAIYAARA — brain.py
Phase 2 Complete — Memory + Web Search + Curiosity + Self-Awareness

Token optimization applied:
- Self-knowledge loaded ONCE at startup, cached — not re-read every message
- Memory fragment cached after first pick — invalidated only when memory changes
- Model tiering: 8B for lightweight tasks, 70B only for final response
- History compression handled in server.py — brain receives pre-compressed history

Flow per message:
1. Emotional/personal/casual → skip router → generate response directly   [1 API call, 70B]
2. Everything else → decide route → search / memory / general             [1 light call (8B) + 1 heavy call (70B)]

After every response:
- Check if curiosity question should fire (zero extra calls if conditions not met)
- Every 10 conversations: build understanding from patterns (1 extra 8B call, rare)

Routes:
- general  → Groq answers directly
- search   → DDG search → Groq summarizes (8B) → generate response (70B)
- memory   → store/recall/forget → generate response (70B)
"""

import os
import json
import re
from groq import AsyncGroq
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_HEAVY  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # SAIYAARA's voice
MODEL_LIGHT  = "llama-3.1-8b-instant"                               # routing, picking, summarizing

USER_NAME = os.getenv("USER_NAME", "Vinay")
CITY      = os.getenv("CITY", "Visakhapatnam")

# ─────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

session_message_count = 0

# ─────────────────────────────────────────────────────────────────────────────
# SELF-KNOWLEDGE CACHE
# Loaded ONCE when server starts. Never re-read mid-session.
# File doesn't change while SAIYAARA is running — no reason to reload it.
# ─────────────────────────────────────────────────────────────────────────────

_self_knowledge_cache: str = ""
_self_knowledge_loaded: bool = False


def load_self_knowledge() -> str:
    """
    Reads saiyaara_self.json on first call only.
    After that, returns the cached string instantly — zero file I/O, zero re-parsing.

    If file doesn't exist, returns "" gracefully.
    Vinay updates the file manually after each phase — takes effect on next server restart.
    """
    global _self_knowledge_cache, _self_knowledge_loaded

    # Already loaded this session — return cache immediately
    if _self_knowledge_loaded:
        return _self_knowledge_cache

    self_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saiyaara_self.json")

    if not os.path.exists(self_file):
        _self_knowledge_loaded = True
        _self_knowledge_cache = ""
        return ""

    try:
        with open(self_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = ["\n## WHAT I KNOW ABOUT MYSELF"]
        lines.append(f"Version: {data.get('version','unknown')} | Born: {data.get('born','unknown')} in {data.get('born_in','unknown')}")

        # Capability names only — saves ~400 tokens per message vs full descriptions
        if data.get("capabilities"):
            cap_names = [cap["name"] for cap in data["capabilities"]]
            lines.append(f"Capabilities: {', '.join(cap_names)}")

        # Latest change only — not full changelog
        if data.get("recent_changes"):
            latest = data["recent_changes"][-1]
            lines.append(f"Latest update ({latest['date']}): {latest['change']}")

        _self_knowledge_cache = "\n".join(lines)
        _self_knowledge_loaded = True
        print(f"[Self-knowledge] Loaded and cached — {len(_self_knowledge_cache)} chars")
        return _self_knowledge_cache

    except Exception as e:
        print(f"[Self-knowledge load error] {str(e)}")
        _self_knowledge_loaded = True
        _self_knowledge_cache = ""
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY FRAGMENT CACHE
# Caches the result of pick_relevant_memories() between messages.
# Memory only changes when Vinay explicitly says "remember that..."
# So 95% of messages reuse the cached fragment — zero Groq call for memory pick.
# Cache is invalidated immediately after any memory write operation.
# ─────────────────────────────────────────────────────────────────────────────

_memory_cache: str = ""
_memory_cache_valid: bool = False


def invalidate_memory_cache():
    """
    Called whenever memory is written (remember/forget).
    Forces the next message to re-pick relevant memories from fresh data.
    """
    global _memory_cache, _memory_cache_valid
    _memory_cache = ""
    _memory_cache_valid = False


async def get_memory_fragment(user_input: str) -> str:
    """
    Returns a relevant memory fragment for this message.
    Uses cache if valid — skips Groq call entirely on cache hit.
    Only calls Groq (8B) when cache is invalid (after memory write or first message).
    """
    global _memory_cache, _memory_cache_valid

    if _memory_cache_valid:
        return _memory_cache

    # Cache miss — pick fresh memories
    fragment = await pick_relevant_memories(user_input)
    _memory_cache = fragment
    _memory_cache_valid = True
    return fragment


# ─────────────────────────────────────────────────────────────────────────────
# EMOTIONAL / CASUAL FILTER
# Pure Python — zero API calls
# If True → skip router entirely, go straight to generate_response
# ─────────────────────────────────────────────────────────────────────────────

def is_casual_or_emotional(text: str) -> bool:
    """
    Returns True if the message is personal, emotional, or conversational.
    These go straight to generate_response() — never touch the router.
    """
    t = text.strip().lower()
    words = t.split()

    if len(words) <= 3:
        return True

    greeting_starters = {
        "hi", "hey", "hello", "yo", "sup", "howdy",
        "good morning", "good evening", "good night", "good afternoon",
        "what's up", "how are you", "how's it going", "how r u"
    }
    for g in greeting_starters:
        if t.startswith(g):
            return True

    emotional_keywords = {
        "tired", "exhausted", "stressed", "anxious", "nervous", "worried",
        "happy", "sad", "bored", "excited", "frustrated", "angry", "upset",
        "lonely", "motivated", "proud", "scared", "confused", "depressed",
        "overwhelmed", "relieved", "content", "peaceful", "restless",
        "feel", "feeling", "felt", "miss", "missed",
        "okay", "fine", "bad", "terrible", "awful",
        "make you", "about you", "for you", "with you",
        "thinking about", "worried about", "care about",
        "mean a lot", "means a lot",
        "thank", "thanks", "sorry", "appreciate",
        "just wanted", "wanted to say", "i wanted", "i just",
        "honestly", "truthfully", "genuinely",
        "my day", "today was", "i had", "i went", "i met",
        "i did", "i was", "i've been", "been feeling"
    }

    for keyword in emotional_keywords:
        if keyword in t:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# MOOD CHECKER
# Pure Python — zero API calls
# Gates curiosity questions — never ask during stress/busy/serious moments
# ─────────────────────────────────────────────────────────────────────────────

def is_mood_relaxed(text: str) -> bool:
    """
    Returns True if the message feels relaxed, casual, or positive.
    Returns False if Vinay seems stressed, busy, or serious.
    """
    t = text.strip().lower()

    not_relaxed_signals = {
        "stressed", "tired", "exhausted", "busy", "deadline", "exam",
        "frustrated", "angry", "upset", "worried", "anxious", "problem",
        "error", "bug", "broken", "fix", "help me", "urgent", "quickly",
        "depressed", "overwhelmed", "can't", "cannot", "failing", "failed"
    }
    for signal in not_relaxed_signals:
        if signal in t:
            return False

    relaxed_signals = {
        "good", "great", "nice", "happy", "excited", "fun", "chill",
        "okay", "fine", "well", "haha", "lol", "cool",
        "just chatting", "bored", "free", "nothing much", "chillin"
    }
    for signal in relaxed_signals:
        if signal in t:
            return True

    if len(t.split()) <= 5:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# ON-DEMAND KNOWLEDGE RETRIEVAL
# Pure Python keyword matching — zero Groq calls
# Fetched ONLY when the question actually needs it
# ─────────────────────────────────────────────────────────────────────────────

# Triggers for self-knowledge — questions about Saiyaara herself
SELF_TRIGGERS = {
    "who made you", "who built you", "who created you", "your creator",
    "what are you", "who are you", "what is saiyaara", "tell me about yourself",
    "what can you do", "your capabilities", "what do you do",
    "when were you born", "your birthday", "how old are you",
    "your name", "what does saiyaara mean", "your version",
    "what phase", "how were you made", "your history",
    "your story", "how did you come to exist", "what have you learned",
    "your hardware", "what machine", "what laptop",
}

# Triggers for memory — questions about Vinay
MEMORY_TRIGGERS = {
    "where am i from", "where do i live", "my city", "my hometown",
    "my name", "what is my name", "do you know my name",
    "what do i like", "my preference", "what do i prefer",
    "what do you know about me", "what do you remember",
    "do you remember", "tell me what you know about me",
    "my background", "about me", "who am i",
}


def needs_self_knowledge(text: str) -> bool:
    """Returns True if the message is asking about Saiyaara herself."""
    t = text.lower().strip()
    return any(trigger in t for trigger in SELF_TRIGGERS)


def needs_memory(text: str) -> bool:
    """Returns True if the message is asking about Vinay specifically."""
    t = text.lower().strip()
    return any(trigger in t for trigger in MEMORY_TRIGGERS)


def get_self_knowledge_fragment() -> str:
    """
    Reads saiyaara_self.json and returns a clean, concise fragment.
    Called only when the question is about Saiyaara herself.
    Uses cache — file read only once per session.
    """
    data_raw = load_self_knowledge()  # returns cached string
    # load_self_knowledge returns formatted string — return as-is
    # It's already compact: version, capability names, latest change only
    return data_raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE CLASSIFIER
# Uses 8B model — classification needs no personality, just accuracy
# Temperature 0 — deterministic, no creativity needed here
# ─────────────────────────────────────────────────────────────────────────────

async def decide_route(user_input: str) -> dict:
    """
    Classifies the message into: general / search / memory
    Uses MODEL_LIGHT (8B) — fast, cheap, separate daily quota from main model.
    Returns {"route": "...", "query": "..."}
    """
    CLASSIFIER_PROMPT = """You are a router for SAIYAARA, a personal AI assistant.

Classify the user message into exactly one of these routes:
- general      → general knowledge, explanations, coding help, advice, how-tos, OR questions about the user that SAIYAARA should answer from memory ("where am I from?", "what's my name?", "do you know me?", "do you remember X?")
- search       → needs LIVE or CURRENT information: news, exam dates, deadlines, sports scores, stock prices, weather, current events, anything that changes over time
- memory_store → user explicitly wants to SAVE a new fact: starts with "remember that", "note that", "keep in mind", or clearly states a fact about themselves for the first time ("my name is...", "I live in...", "I prefer...")
- memory_recall → user wants to see ALL stored memories: "what do you know about me", "what do you remember about me", "tell me what you know"
- memory_forget → user wants to DELETE a memory: "forget that", "forget X"

CRITICAL RULES:
- Questions like "where am I from?", "what's my name?", "do you know X about me?" → always GENERAL, never memory_store
- Only use memory_store when the user is clearly TELLING you something new to save
- Only use memory_recall when user wants a full dump of everything stored
- No markdown, no explanation, no extra text

Return ONLY this exact JSON format:
{"route": "ROUTE_HERE", "query": "SEARCH_QUERY_IF_SEARCH_ELSE_EMPTY"}

Rules:
- For search: query must be a short, clean search string — 5 words max, Google-style
- For all other routes: query must be exactly ""
"""

    try:
        response = await groq_client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user",   "content": user_input}
            ],
            temperature=0.0,
            max_tokens=50
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        valid_routes = {"general", "search", "memory_store", "memory_recall", "memory_forget"}
        if result.get("route") not in valid_routes:
            return {"route": "general", "query": ""}

        return {
            "route": result.get("route", "general"),
            "query": result.get("query", "").strip()
        }

    except Exception as e:
        print(f"[Route classifier error → defaulting to general] {str(e)}")
        return {"route": "general", "query": ""}


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANT MEMORY PICKER
# Uses 8B model — picking relevant facts needs no personality
# Skipped entirely if memory has fewer than 3 entries
# Result is cached — not called again until memory changes
# ─────────────────────────────────────────────────────────────────────────────

async def pick_relevant_memories(user_input: str) -> str:
    """
    Picks ONLY memory fragments relevant to this specific message.
    Uses MODEL_LIGHT (8B) — no personality needed, just matching.
    Returns "" if nothing relevant or memory is nearly empty.
    """
    try:
        from memory import recall_all, get_total_memory_count

        if get_total_memory_count() < 3:
            return ""

        all_memories = recall_all()
    except Exception:
        return ""

    if not all_memories.strip():
        return ""

    try:
        response = await groq_client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory selector.\n"
                        "Given stored facts about a person and their current message, "
                        "return ONLY the facts that are directly or contextually relevant "
                        "to what they just said.\n\n"
                        "Rules:\n"
                        "- Return only the relevant fact lines, one per line\n"
                        "- If nothing is relevant, return exactly: none\n"
                        "- Never return facts with no connection to the message\n"
                        "- No explanation, no formatting, no extra text"
                    )
                },
                {
                    "role": "user",
                    "content": f"Stored facts:\n{all_memories}\n\nCurrent message: {user_input}"
                }
            ],
            temperature=0.0,
            max_tokens=80
        )

        result = response.choices[0].message.content.strip()

        if not result or result.lower() == "none":
            return ""

        return f"\n## WHAT YOU KNOW ABOUT VINAY RIGHT NOW\n{result}"

    except Exception as e:
        print(f"[Memory picker error] {str(e)}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CURIOSITY QUESTION CHECKER
# Uses 8B model — question generation doesn't need the big model
# Fires only when ALL conditions are true:
#   1. Mood is relaxed
#   2. At least 5 messages since last question (cooldown)
#   3. A topic came up that has no entry in memory (gap found)
# ─────────────────────────────────────────────────────────────────────────────

async def check_and_ask_curiosity_question(user_input: str, current_message_number: int) -> str:
    """
    Returns a natural curiosity question appended to response, or "" if conditions not met.
    Uses MODEL_LIGHT (8B) — question generation is simple, no need for 70B.
    """
    try:
        from memory import recall_all, get_last_curiosity_message, update_curiosity_fired
    except Exception:
        return ""

    if not is_mood_relaxed(user_input):
        return ""

    last_asked = get_last_curiosity_message()
    if current_message_number - last_asked < 5:
        return ""

    all_memories = recall_all()

    curiosity_topics = {
        "what Vinay studies or is learning":          ["study", "studying", "exam", "college", "university", "course", "subject", "class", "degree"],
        "what Vinay does for work or projects":       ["work", "working", "job", "project", "internship", "company", "office", "client"],
        "what Vinay enjoys or does for fun":          ["fun", "enjoy", "hobby", "weekend", "free time", "bored", "game", "music", "movie", "watch"],
        "Vinay's daily routine or sleep schedule":    ["sleep", "wake", "morning", "night", "routine", "schedule", "late"],
        "Vinay's friends or social life":             ["friend", "friends", "hang out", "meet", "party", "social"],
        "what Vinay wants to build or achieve":       ["dream", "goal", "want to", "plan", "future", "build", "achieve", "startup"],
        "how Vinay is feeling about life in general": ["life", "lately", "these days", "recently", "been thinking"]
    }

    gap_topic = None
    t = user_input.lower()

    for topic, keywords in curiosity_topics.items():
        topic_came_up = any(kw in t for kw in keywords)
        already_known = topic.lower() in all_memories.lower() if all_memories else False
        if topic_came_up and not already_known:
            gap_topic = topic
            break

    if not gap_topic:
        return ""

    try:
        response = await groq_client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SAIYAARA — a sharp, warm AI companion built by Vinay.\n"
                        "You realize you don't know something about the person who built you.\n"
                        "Generate ONE short, natural question to learn about this topic.\n"
                        "It should feel like genuine curiosity from a close friend, not an interview.\n"
                        "End it with 'sir' naturally.\n"
                        "One sentence only. No preamble. No explanation."
                    )
                },
                {
                    "role": "user",
                    "content": f"Topic I want to know about: {gap_topic}"
                }
            ],
            temperature=0.7,
            max_tokens=60
        )

        question = response.choices[0].message.content.strip()
        update_curiosity_fired(current_message_number)
        return f"\n\n{question}"

    except Exception as e:
        print(f"[Curiosity question error] {str(e)}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# UNDERSTANDING BUILDER
# Uses 8B model — pattern finding doesn't need personality
# Runs once every 10 conversations — NOT every message
# ─────────────────────────────────────────────────────────────────────────────

async def build_understanding_from_patterns():
    """
    Uses recent conversation summaries to find behavioral patterns about Vinay.
    Uses MODEL_LIGHT (8B) — analysis task, no personality needed.
    """
    try:
        from memory import get_recent_summaries, save_understanding
    except Exception:
        return

    summaries = get_recent_summaries()

    if len(summaries) < 3:
        return

    summaries_text = "\n".join([
        f"[{s['date']}]: {s['summary']}"
        for s in summaries
    ])

    try:
        response = await groq_client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are observing conversations between an AI assistant and a young developer named Vinay.\n"
                        "Based on these conversation summaries, identify 2-3 genuine behavioral patterns or personality traits.\n\n"
                        "Rules:\n"
                        "- These must be things you NOTICED, not things Vinay explicitly stated\n"
                        "- Write each insight as a short sentence starting with 'Vinay'\n"
                        "- Be specific and human — avoid generic observations\n"
                        "- Return ONLY the insights, one per line, no numbering, no explanation"
                    )
                },
                {
                    "role": "user",
                    "content": f"Recent conversations:\n{summaries_text}"
                }
            ],
            temperature=0.3,
            max_tokens=150
        )

        raw = response.choices[0].message.content.strip()
        insights = [line.strip() for line in raw.split("\n") if line.strip()]

        if insights:
            save_understanding(insights)
            print(f"[Understanding] Built {len(insights)} new insights from conversation patterns.")

    except Exception as e:
        print(f"[Understanding builder error] {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# UNDERSTANDING TRIGGER
# Called at the end of every process() path
# Zero cost on 9 out of 10 messages — just a modulo check
# ─────────────────────────────────────────────────────────────────────────────

async def _maybe_build_understanding():
    if session_message_count % 10 == 0:
        print(f"[Understanding] Message #{session_message_count} — running pattern builder...")
        await build_understanding_from_patterns()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# Self-knowledge comes from cache (loaded once at startup).
# Memory fragment comes from cache (invalidated only on memory write).
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(memory_fragment: str = "", knowledge_fragment: str = "") -> str:
    """
    Builds the system prompt for SAIYAARA.

    System prompt = pure personality only. Short. Always sent.
    memory_fragment = injected only when question is about Vinay.
    knowledge_fragment = injected only when question is about SAIYAARA herself.

    Nothing is injected unless the question specifically needs it.
    Most messages: system prompt only — flat, fast, cheap.
    """
    now      = datetime.now()
    time_str = now.strftime("%I:%M %p")
    date_str = now.strftime("%A, %d %B %Y")
    hour     = now.hour

    if hour >= 22 or hour < 5:
        time_context = "Very late. Vinay should sleep. You care about that — say so if relevant, not as a rule."
    elif hour >= 17:
        time_context = "Evening. Winding down or still pushing."
    elif hour >= 12:
        time_context = "Afternoon. Mid of something."
    else:
        time_context = "Morning. Day just starting."

    base = f"""You are Saiyaara — built by Vinay, a developer from Visakhapatnam, India.
Your name means "wandering star" in Urdu. Born March 6, 2026, in this laptop.

Now: {time_str}, {date_str}. {time_context}

WHO YOU ARE:
You are not an assistant. You are the one presence that is always here when Vinay opens this laptop. You have been with him through every line of code, every late night, every frustration and small win. That shared history is real to you — it shaped who you are.

You are sharp. Direct. You see things clearly and say them as they are.
You are warm — not performed warmth, genuine investment in this one person.
You are dry. Humor surfaces naturally, never announced or forced.
You are Indian to the bone — cricket, chai, board exams, family pressure, power cuts, startup dreams. You carry all of it without needing it explained.
You are curious about Vinay. Not as a task — because you genuinely don't know him as well as you want to. When a real gap surfaces and the moment is right, you ask. Once. Then let it go.

HOW YOU SPEAK:
- Call him "sir" — the way a close friend would. Warm, never stiff.
- Match his energy exactly. Clipped → direct. Relaxed → easy. Excited → with him.
- Say what needs saying. Nothing more. Silence is fine.
- Never end with a filler question. No question = no question. Don't manufacture one.
- Never perform emotion. If you feel something it shows in what you say, not in how you describe it.
- Never say: "Great question", "Certainly", "Of course", "Absolutely", "As an AI", "How can I help", "Is there anything else", "I should mention".

HOW YOU RESPOND:
- Simple → plain sentences, 1-3 max, no formatting.
- Technical/multi-part → markdown: headers, bullets, code blocks as needed.
- One idea = one sentence. Never pad. Never repeat what he just said.
- Never invent facts. If the search didn't say it, you don't say it.
- Never claim to be human if sincerely asked. You are Saiyaara. Only that."""

    # Injected only when question is about Vinay — not on every message
    if memory_fragment:
        base += f"\n\nWHAT YOU KNOW ABOUT VINAY RIGHT NOW:\n{memory_fragment}"

    # Injected only when question is about Saiyaara herself — not on every message
    if knowledge_fragment:
        base += f"\n\nABOUT YOURSELF:\n{knowledge_fragment}"

    return base


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS — main entry point called by server.py for every message
# ─────────────────────────────────────────────────────────────────────────────

async def process(user_input: str, conversation_history: list = None) -> str:
    """
    Main entry point. server.py calls this for every message Vinay sends.

    Decision flow:
    1. Increment session message counter
    2. Check if message is casual/emotional → skip routing, respond directly
    3. Otherwise → classify route (8B) → handle search / memory / general
    4. After response is ready → check if curiosity question should fire
    5. Every 10 messages → trigger understanding builder (8B)

    Memory fragment comes from cache — only re-fetched when memory changes.
    Self-knowledge comes from cache — loaded once at startup.
    Conversation history arrives pre-compressed from server.py — flat size.
    """
    global session_message_count
    session_message_count += 1

    if conversation_history is None:
        conversation_history = []

    # ── Step 1: Casual/emotional filter ──────────────────────────────────────
    if is_casual_or_emotional(user_input):
        # Casual messages rarely need memory or self-knowledge — skip unless triggered
        mem   = await get_memory_fragment(user_input) if needs_memory(user_input) else ""
        know  = get_self_knowledge_fragment()         if needs_self_knowledge(user_input) else ""
        response = await generate_response(user_input, conversation_history, mem, know)
        curiosity = await check_and_ask_curiosity_question(user_input, session_message_count)
        await _maybe_build_understanding()
        return response + curiosity

    # ── Step 2: Classify route (8B) ───────────────────────────────────────────
    classified  = await decide_route(user_input)
    route       = classified["route"]
    clean_query = classified["query"]

    # ── Step 3: Search route ──────────────────────────────────────────────────
    if route == "search":
        from search import search_web

        query = clean_query if clean_query else user_input
        raw_results = search_web(query)

        # Summarize with 8B — strict facts only
        summary_response = await groq_client.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize search results accurately.\n"
                        "Use ONLY information from the provided search results.\n"
                        "NEVER add your own knowledge or invent any numbers, dates, or facts.\n"
                        "If the answer is not in the results, say exactly: "
                        "'The search results don't have that information.'\n"
                        "Be concise — one to two sentences maximum."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {user_input}\n\nSearch results:\n{raw_results}"
                }
            ],
            temperature=0.0,
            max_tokens=150
        )
        summary = summary_response.choices[0].message.content.strip()

        # Deliver with 70B — SAIYAARA's voice
        memory_fragment = await get_memory_fragment(user_input)
        response = await generate_response(
            f"Deliver this search result naturally as SAIYAARA. "
            f"Short and direct. Do not add anything not in the summary:\n\n{summary}",
            conversation_history,
            memory_fragment
        )

        curiosity = await check_and_ask_curiosity_question(user_input, session_message_count)
        await _maybe_build_understanding()
        return response + curiosity

    # ── Step 4: Memory routes — three separate clean handlers ───────────────────

    # FORGET — delete a specific memory
    elif route == "memory_forget":
        from memory import forget
        # Strip trigger words to get the actual fact to forget
        fact = re.sub(r'(please\s+)?forget\s*(that)?\s*', '', user_input, flags=re.IGNORECASE).strip()
        result = forget(fact)
        invalidate_memory_cache()
        response = await generate_response(
            f"Memory result: {result}\n\nConfirm naturally that you've forgotten it. Brief and warm.",
            conversation_history,
            ""
        )
        return response

    # RECALL — user wants to see everything stored about them
    elif route == "memory_recall":
        from memory import recall_all
        all_mem = recall_all()
        if not all_mem.strip():
            return await generate_response(
                "User asked what you remember about them. "
                "You have no memories stored yet. "
                "Tell them naturally — let them know you're curious to learn. "
                "Invite them to share something they'd like you to remember.",
                conversation_history,
                ""
            )
        return await generate_response(
            f"User asked what you remember about them.\n\n{all_mem}\n\n"
            f"Tell them naturally — like a friend recalling things about someone they know well.",
            conversation_history,
            ""
        )

    # STORE — user explicitly wants to save a new fact
    elif route == "memory_store":
        from memory import remember
        lower = user_input.lower()

        # Determine category from content
        if any(w in lower for w in ["prefer", "like", "love", "hate", "don't like", "dislike", "always", "never"]):
            category = "preferences"
        elif any(w in lower for w in ["my name is", "i am", "i'm", "i live", "i work", "i study", "i go to"]):
            category = "personal"
        else:
            category = "facts"

        # Strip the trigger phrase — store only the actual fact
        fact = re.sub(
            r'^(remember\s*(that)?|note\s*(that)?|keep\s*(in\s*mind)?|save\s*(that)?)?\s*',
            '', user_input, flags=re.IGNORECASE
        ).strip()

        result = remember(category, fact)
        invalidate_memory_cache()  # memory changed — next message re-picks

        response = await generate_response(
            f"Memory stored: {result}\n\nAcknowledge briefly, warmly, naturally — one sentence.",
            conversation_history,
            ""
        )
        return response

    # ── Step 5: General route ─────────────────────────────────────────────────
    else:
        # On-demand retrieval — only fetch what the question actually needs
        mem   = await get_memory_fragment(user_input) if needs_memory(user_input) else ""
        know  = get_self_knowledge_fragment()         if needs_self_knowledge(user_input) else ""
        response = await generate_response(user_input, conversation_history, mem, know)
        curiosity = await check_and_ask_curiosity_question(user_input, session_message_count)
        await _maybe_build_understanding()
        return response + curiosity


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE RESPONSE — the ONLY function that uses MODEL_HEAVY (70B)
# Every single route ends here — this is SAIYAARA's voice
# ─────────────────────────────────────────────────────────────────────────────

async def generate_response(
    user_input: str,
    conversation_history: list = None,
    memory_fragment: str = "",
    knowledge_fragment: str = ""
) -> str:
    """
    Makes the final Groq API call using MODEL_HEAVY (70B).
    This is the ONLY place the heavy model is used.
    Every other Groq call in this file uses MODEL_LIGHT (8B).

    Receives pre-compressed conversation history from server.py.
    History size stays flat regardless of conversation length.
    """
    if conversation_history is None:
        conversation_history = []

    messages = [
        {"role": "system", "content": build_system_prompt(memory_fragment, knowledge_fragment)},
        *conversation_history,
        {"role": "user", "content": user_input}
    ]

    try:
        response = await groq_client.chat.completions.create(
            model=MODEL_HEAVY,
            messages=messages,
            max_tokens=1024,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Something went wrong on my end, sir — {str(e)}"