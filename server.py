"""
SAIYAARA — server.py
Pure FastAPI WebSocket bridge. Zero logic lives here.
Receives messages from browser → calls brain → sends response back.
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from brain import process, think

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

# ── Conversation History ──────────────────────────────────────────────────────
conversation_history = []

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

# ── Opening greeting based on time ───────────────────────────────────────────
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
    return await think(prompt, [])

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global conversation_history

    await websocket.accept()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected")

    # Send connection confirmation
    await websocket.send_json({
        "type": "connected",
        "message": "SAIYAARA is online."
    })

    # Send opening greeting
    opening = await get_opening_greeting()
    await websocket.send_json({"type": "response", "text": opening})

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except:
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

                reply = await process(text, conversation_history)
                response = {"type": "response", "text": reply}

                # Update conversation history
                conversation_history.append({"role": "user", "content": text})
                spoken = response.get("text", "")
                if spoken:
                    conversation_history.append({"role": "assistant", "content": spoken})

                # Trim to last 20 exchanges
                if len(conversation_history) > 40:
                    conversation_history = conversation_history[-40:]

                # Terminal log
                log_text = response.get("text") or response["type"]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SAIYAARA [{response['type']}]: {str(log_text)[:80]}...")

                await websocket.send_json(response)

    except WebSocketDisconnect:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Disconnected")
        conversation_history.clear()

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  SAIYAARA is running → http://{HOST}:{PORT}")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="warning")