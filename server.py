"""
SAIYAARA — server.py
Pure FastAPI WebSocket bridge. Zero logic lives here except history compression.

History compression (the core token fix):
- Keep last 6 messages (3 exchanges) raw — immediate context
- Everything older → compressed into one summary line via Groq (8B)
- Input size to brain.py stays FLAT regardless of conversation length
- Without this: message 30 sends ~2000 history tokens. With this: always ~200.
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()

from brain import process, generate_response

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SAIYAARA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 2004))

# ── Groq client for history compression (8B only — lightweight task) ──────────
_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_LIGHT = "llama-3.1-8b-instant"

# ── Conversation History ──────────────────────────────────────────────────────
# Full raw history — kept here in server.py
# Compressed version is what gets sent to brain.py
conversation_history: list = []

# ── History compression constants ────────────────────────────────────────────
# Keep this many recent messages raw (passed as-is to brain)
# Everything before this window gets compressed into one summary
RAW_WINDOW      = 6   # last 6 messages = last 3 exchanges, always raw
COMPRESS_AFTER  = 10  # start compressing once history exceeds this many messages


# ── History Compressor ────────────────────────────────────────────────────────

async def compress_history(history: list) -> list:
    """
    Compresses older conversation history into a single summary message.
    Keeps the last RAW_WINDOW messages intact for immediate context.

    Structure returned:
    [
      {"role": "system", "content": "Earlier in this conversation: <summary>"},
      ... last RAW_WINDOW messages raw ...
    ]

    This keeps the input to brain.py at a flat, predictable size
    no matter how long the conversation runs.

    Only fires when history exceeds COMPRESS_AFTER messages.
    Below that threshold, returns history unchanged — no Groq call needed.
    """
    if len(history) <= COMPRESS_AFTER:
        return history  # short conversation — no compression needed yet

    older   = history[:-RAW_WINDOW]   # everything except the last RAW_WINDOW messages
    recent  = history[-RAW_WINDOW:]   # last RAW_WINDOW messages — always kept raw

    # Format older messages for summarization
    older_text = "\n".join([
        f"{'Vinay' if m['role'] == 'user' else 'SAIYAARA'}: {m['content']}"
        for m in older
    ])

    try:
        response = await _groq.chat.completions.create(
            model=MODEL_LIGHT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You summarize a conversation between Vinay and SAIYAARA (his AI assistant).\n"
                        "Write a single short paragraph — 2 to 3 sentences maximum.\n"
                        "Cover: main topics discussed, any decisions made, any important context.\n"
                        "Write in past tense. Be factual and concise. No filler."
                    )
                },
                {
                    "role": "user",
                    "content": f"Conversation to summarize:\n{older_text}"
                }
            ],
            temperature=0.0,
            max_tokens=120
        )

        summary = response.choices[0].message.content.strip()

        # Return: one summary message + recent raw messages
        compressed = [
            {
                "role": "system",
                "content": f"Earlier in this conversation: {summary}"
            }
        ] + recent

        print(f"[History] Compressed {len(older)} messages → 1 summary. Keeping {len(recent)} raw.")
        return compressed

    except Exception as e:
        # If compression fails for any reason, fall back to just the recent window
        # Better to lose older context than to crash or send a huge history
        print(f"[History compression error — using recent window only] {str(e)}")
        return recent


# ── Serve Main UI ─────────────────────────────────────────────────────────────
@app.get("/")
async def serve_ui():
    return FileResponse("saiyaara.html")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "running",
        "time": datetime.now().strftime("%I:%M %p")
    }


# ── Opening Greeting ──────────────────────────────────────────────────────────
async def get_opening_greeting() -> str:
    hour = datetime.now().hour
    if hour >= 22 or hour < 5:
        prompt = "Vinay just opened his laptop very late at night. Greet him with genuine concern. Short, warm, real. No more than 2 sentences."
    elif hour >= 17:
        prompt = "Vinay just opened his laptop in the evening. Greet him naturally, check in on his day. Short and warm. No more than 2 sentences."
    elif hour >= 12:
        prompt = "Vinay just opened his laptop in the afternoon. Greet him casually and naturally. One sentence."
    else:
        prompt = "Vinay just opened his laptop in the morning. Greet him with energy, wish him a good day. One sentence."
    return await generate_response(prompt, [])


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global conversation_history

    await websocket.accept()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected")

    await websocket.send_json({
        "type": "connected",
        "message": "SAIYAARA is online."
    })

    opening = await get_opening_greeting()
    await websocket.send_json({"type": "response", "text": opening})

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except Exception:
                data = {"type": "message", "text": raw}

            msg_type = data.get("type", "message")
            text     = data.get("text", "").strip()

            # ── Ping ──────────────────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # ── Message ───────────────────────────────────────────────────────
            if msg_type == "message" and text:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Vinay: {text}")
                await websocket.send_json({"type": "thinking"})

                # Compress history before sending to brain
                # This keeps the input size flat — core of the token fix
                compressed = await compress_history(conversation_history)

                reply = await process(text, compressed)
                response = {"type": "response", "text": reply}

                # Store full raw exchange in local history
                # (compression happens on read, not on write)
                conversation_history.append({"role": "user",      "content": text})
                conversation_history.append({"role": "assistant", "content": reply})

                # Hard cap on raw history to prevent unbounded memory usage
                # Even with compression, we don't want the raw list growing forever
                if len(conversation_history) > 60:
                    conversation_history = conversation_history[-60:]

                log_text = response.get("text") or response["type"]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SAIYAARA: {str(log_text)[:80]}...")

                await websocket.send_json(response)

    except WebSocketDisconnect:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Disconnected")
        conversation_history.clear()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  SAIYAARA is running → http://{HOST}:{PORT}")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="warning")