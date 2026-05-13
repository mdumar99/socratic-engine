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
User query
│
▼
Debate Orchestrator
│
├── Round 0 ──▶ Proposer   (Phi-3 mini, CUDA) ──▶ hypothesis
├── Round 1 ──▶ Critic     (Phi-3 mini, CUDA) ──▶ critique
├── Round 2 ──▶ Retriever  (Phi-3 mini, CUDA) ──▶ evidence
└── Round 3 ──▶ Synthesizer(Phi-3 mini, CUDA) ──▶ final answer
All agents share one loaded model (llama.cpp C API)
Each agent has its own llama_context — no VRAM duplication
C++ Dispatcher ←──── Knowledge Graph (RocksDB + BFS)
│                   BFS traversal < 2ms
└── Lock-free SPSC ring buffers (one per agent)
Zero mutex overhead, cache-line aligned
## Verified performance

- Model: Phi-3 mini Q4_K_M (2.3GB, 3.8B params)
- GPU: NVIDIA Quadro RTX 5000 16GB (CUDA 12.6)
- Full 4-round debate: ~63 seconds end-to-end
- Model loaded once, 4 contexts share weights in VRAM
- 3/3 unit tests passing, CI green

## Stack

| Layer | Technology |
|---|---|
| Agent inference | llama.cpp C API (GGUF, CUDA backend) |
| Debate orchestration | C++ DebateOrchestrator |
| IPC | Lock-free SPSC ring buffers |
| Knowledge graph | RocksDB + custom BFS traversal |
| Build | CMake 3.20+, GCC 13, CUDA 12.6 |
| Python tooling | huggingface-hub, faiss-cpu, pytest, black |

## Project structure
socratic-engine/
├── cpp/
│   ├── ipc/              # Lock-free SPSC ring buffer + AgentJob
│   ├── dispatcher/       # Dispatcher + DebateOrchestrator
│   ├── agents/           # AgentRunner + LlamaEngine (C API)
│   └── knowledge_graph/  # RocksDB graph + BFS traversal
├── python/               # CLI, KG builder, eval tooling (WIP)
├── tests/unit/           # C++ unit tests (cmake + ctest)
├── docs/                 # Setup guide + architecture decisions
└── scripts/              # bootstrap.sh, dev helpers
## Setup

See [`docs/setup.md`](docs/setup.md)

## Branch strategy

- `main` — protected, always green, release-ready
- `dev` — integration branch, all PRs target here
- `feat/*` — feature branches, PR into dev
