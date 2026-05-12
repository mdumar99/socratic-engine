# Architecture decision records

## ADR-001 — Use llama.cpp over Python inference
**Status**: accepted  
**Reason**: True multi-threading, no GIL, SIMD-optimized GGML kernels, zero-copy CUDA buffers.

## ADR-002 — RocksDB for knowledge graph backing store
**Status**: accepted  
**Reason**: Already installed on target system, LSM-tree writes suit our append-heavy KG update pattern.

## ADR-003 — Lock-free SPSC ring buffers for IPC
**Status**: accepted  
**Reason**: Zero mutex overhead between dispatcher and agents on same machine.
