#!/usr/bin/env python3
"""
Socratic Engine — FastAPI server with token streaming
"""

import asyncio
import subprocess
import json
import os
import sys
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

PROJECT_ROOT = Path(__file__).parent.parent
BINARY = PROJECT_ROOT / "build/cpp/dispatcher/debate_runner"
LLAMA_LIBS = Path.home() / "llama.cpp/build/bin"
STATIC_DIR = Path(__file__).parent / "static"

# Persistent KG path — set SE_KG_PATH env var to override
KG_PATH = os.environ.get("SE_KG_PATH", str(Path.home() / ".socratic_engine" / "kg"))

app = FastAPI(title="Socratic Engine", version="0.2.0")


async def stream_debate(query: str, tokens: int = 400) -> AsyncIterator[dict]:
    """Run debate_runner and yield parsed JSON events as they arrive."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_LIBS)

    cmd = [str(BINARY), "--query", query, "--tokens", str(tokens), "--kg", KG_PATH]

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


@app.post("/kg/add")
async def kg_add(body: dict):
    """Add text content to the persistent KG."""
    text = body.get("text", "").strip()
    label = body.get("label", "text")
    base_id = int(body.get("base_id", 1000))

    if not text:
        return {"status": "error", "message": "No text provided"}

    ingest = PROJECT_ROOT / "build/cpp/knowledge_graph/kg_ingest"
    if not ingest.exists():
        return {"status": "error", "message": "kg_ingest binary not found"}

    # Use sources to chunk text into nodes
    sys.path.insert(0, str(PROJECT_ROOT))
    from python.kg_builder.sources import nodes_from_string
    from python.kg_builder.builder import KGBuilder

    nodes = nodes_from_string(text, label=label, base_id=base_id)
    kg = KGBuilder(KG_PATH)
    written = kg.add_nodes_batch(nodes)

    # Connect new nodes to root node (id=1) so BFS can reach them
    for node in nodes:
        kg.add_edge(1, node["id"], relation="RELATED_TO", weight=0.8)

    count = kg.count()

    return {
        "status": "ok",
        "nodes_written": written,
        "total_nodes": count,
        "kg_path": KG_PATH,
    }


@app.get("/kg/info")
async def kg_info():
    """Return KG path and node count."""
    from pathlib import Path as P

    ingest = PROJECT_ROOT / "build/cpp/knowledge_graph/kg_ingest"
    count = 0
    if ingest.exists() and P(KG_PATH).exists():
        result = subprocess.run(
            [str(ingest), "--kg", KG_PATH],
            input='{"op":"count"}\n',
            capture_output=True,
            text=True,
        )
        import json as _json

        for line in result.stdout.strip().splitlines():
            try:
                d = _json.loads(line)
                if d.get("status") == "ok":
                    count = d.get("count", 0)
            except Exception:
                pass
    return {"kg_path": KG_PATH, "node_count": count}


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
