# Socratic Engine

A distributed multi-agent LLM system where small models (Phi-3 mini, 3.8B) achieve
large-model quality through structured 4-round Socratic debate, C++ orchestration,
and knowledge graph-grounded context injection.

## How it works

A user query enters the system and triggers a structured debate across 4 specialist
agents. Each agent has a distinct role and reads the previous round's output before
generating its own — the system argues with itself to produce a better answer than
any single model call could.

**Round 0 — Proposer**: generates an initial hypothesis grounded in KG context  
**Round 1 — Critic**: identifies flaws, gaps, and unsupported claims  
**Round 2 — Retriever**: extracts supporting or refuting evidence from the knowledge graph  
**Round 3 — Synthesizer**: merges all three into a final, refined answer  

## Architecture

```text
User query
    │
    ▼
Debate Orchestrator
│
├── Round 0 ──▶ Proposer    (Phi-3 mini, CUDA) ──▶ hypothesis
├── Round 1 ──▶ Critic      (Phi-3 mini, CUDA) ──▶ critique
├── Round 2 ──▶ Retriever   (Phi-3 mini, CUDA) ──▶ evidence
└── Round 3 ──▶ Synthesizer (Phi-3 mini, CUDA) ──▶ final answer
│
▼
merged answer
(streamed live to browser via WebSocket)
All agents share one loaded model (llama.cpp C API)
Each agent has its own llama_context — no VRAM duplication
C++ Dispatcher ←──── Knowledge Graph (RocksDB + BFS, < 2ms)
│
└── Lock-free SPSC ring buffers (one per agent)
Zero mutex overhead, cache-line aligned
## Web UI
Browser  ──WebSocket──▶  FastAPI server  ──subprocess──▶  debate_runner (C++)
│
token events stream
back as generated

Ask a question, watch 4 agents debate live — tokens appear as they are generated.

## Verified performance

- Model: Phi-3 mini Q4_K_M (2.3GB, 3.8B params)
- GPU: NVIDIA Quadro RTX 5000 16GB (CUDA 12.6)
- Full 4-round debate: ~80s end-to-end
- Token streaming: each token fires a WebSocket event as generated
- Model loaded once, 4 contexts share weights in VRAM
- 3/3 unit tests passing, CI green

## Stack

| Layer | Technology |
|---|---|
| Agent inference | llama.cpp C API (GGUF, CUDA backend) |
| Debate orchestration | C++ DebateOrchestrator |
| Token streaming | generate_stream() callback → WebSocket |
| IPC | Lock-free SPSC ring buffers |
| Knowledge graph | RocksDB + custom BFS traversal |
| API server | FastAPI + WebSockets (Python) |
| Web UI | Vanilla JS, single-file, dark theme |
| Build | CMake 3.20+, GCC 13, CUDA 12.6 |
| Python tooling | huggingface-hub, faiss-cpu, pytest, black |

## Project structure

```text
socratic-engine/
├── cpp/
│   ├── ipc/               # Lock-free SPSC ring buffer + AgentJob
│   ├── dispatcher/        # Dispatcher + DebateOrchestrator + debate_runner
│   ├── agents/            # AgentRunner + LlamaEngine (C API, token streaming)
│   └── knowledge_graph/   # RocksDB graph + BFS traversal
├── python/
│   ├── static/            # Web UI (index.html)
│   ├── server.py          # FastAPI + WebSocket server
│   ├── cli.py             # Terminal CLI with colored output
│   └── requirements.txt
├── tests/unit/            # C++ unit tests (cmake + ctest)
├── docs/                  # Setup guide + architecture decisions
└── scripts/               # bootstrap.sh, dev helpers

## Running

**Web UI:**
```bash
source python/venv/bin/activate
uvicorn python.server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

**CLI:**
```bash
source python/venv/bin/activate
python python/cli.py "What causes lightning?"
python python/cli.py --interactive
```

## Setup

See [docs/setup.md](docs/setup.md)

## Branch strategy

- `main` — protected, always green, release-ready
- `dev` — integration branch, all PRs target here
- `feat/*` — feature branches, PR into dev
