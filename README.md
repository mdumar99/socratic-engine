# Socratic Engine

A distributed multi-agent LLM system where small models (Phi-3 mini, 3.8B) achieve
large-model quality through structured debate, C++ orchestration, and knowledge
graph-grounded context injection.

## How it works

Four specialist agents (Proposer, Critic, Retriever, Synthesizer) run in parallel,
communicating via lock-free SPSC ring buffers managed by a C++ dispatcher. A
RocksDB-backed knowledge graph prunes context to ~512 tokens per agent before
inference, so each small model reasons about exactly what it needs to.
## Architecture
User query
│
▼
C++ Dispatcher  ←──  Knowledge Graph (RocksDB + BFS)
│
├──▶ Proposer agent   (Phi-3 mini, CUDA)
├──▶ Critic agent     (Phi-3 mini, CUDA)
├──▶ Retriever agent  (Phi-3 mini, CUDA)
└──▶ Synthesizer      (Phi-3 mini, CUDA)
│
▼
Merged answer (grounded, critiqued, confidence-scored)

## Stack

| Layer | Technology |
|---|---|
| Agent inference | llama.cpp (GGUF, CUDA backend) |
| IPC | Lock-free SPSC ring, POSIX shared memory |
| Knowledge graph | RocksDB + custom BFS traversal |
| Embedding similarity | FAISS |
| Build | CMake 3.20+, GCC 13, CUDA 12 |
| Python tooling | huggingface-hub, pytest, black |

## Setup

See [`docs/setup.md`](docs/setup.md)

## Branch strategy

- `main` — protected, always green
- `dev` — integration branch
- `feat/*` — feature branches, PR into dev
