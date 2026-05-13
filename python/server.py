#!/usr/bin/env python3
"""
Socratic Engine — FastAPI server with token streaming
"""
import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

PROJECT_ROOT = Path(__file__).parent.parent
BINARY = PROJECT_ROOT / "build/cpp/dispatcher/debate_runner"
LLAMA_LIBS = Path.home() / "llama.cpp/build/bin"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Socratic Engine", version="0.2.0")


async def stream_debate(query: str, tokens: int = 400) -> AsyncIterator[dict]:
    """Run debate_runner and yield parsed JSON events as they arrive."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_LIBS)

    cmd = [str(BINARY), "--query", query, "--tokens", str(tokens)]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )

    async for raw_line in proc.stdout:
        line = raw_line.decode().strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            yield event
        except json.JSONDecodeError:
            continue

    await proc.wait()


@app.websocket("/ws/debate")
async def debate_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        query = data.get("query", "").strip()
        tokens = int(data.get("tokens", 250))

        if not query:
            await websocket.send_json({"event": "error", "message": "Empty query"})
            return

        if not BINARY.exists():
            await websocket.send_json(
                {"event": "error", "message": "debate_runner binary not found"}
            )
            return

        async for event in stream_debate(query, tokens):
            await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "binary_exists": BINARY.exists(),
        "llama_libs": str(LLAMA_LIBS),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Socratic Engine API</h1><p>UI not built yet.</p>")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
