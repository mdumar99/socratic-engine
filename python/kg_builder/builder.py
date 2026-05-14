"""
KGBuilder — writes nodes and edges to RocksDB via kg_ingest binary.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
INGEST_BIN = PROJECT_ROOT / "build/cpp/knowledge_graph/kg_ingest"


class KGBuilder:
    def __init__(self, kg_path: str):
        self.kg_path = kg_path
        if not INGEST_BIN.exists():
            raise RuntimeError(
                f"kg_ingest binary not found at {INGEST_BIN}\n"
                "Run: cmake --build build --target kg_ingest"
            )

    def _send(self, ops: list[dict]) -> list[dict]:
        """Send a batch of ops to kg_ingest, return responses."""
        lines = "\n".join(json.dumps(op) for op in ops) + "\n"
        result = subprocess.run(
            [str(INGEST_BIN), "--kg", self.kg_path],
            input=lines,
            capture_output=True,
            text=True,
        )
        responses = []
        for line in result.stdout.strip().splitlines():
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return responses

    def add_node(
        self,
        node_id: int,
        label: str,
        content: str,
        confidence: float = 0.9,
    ) -> bool:
        responses = self._send(
            [
                {
                    "op": "node",
                    "id": node_id,
                    "label": label,
                    "content": content,
                    "confidence": confidence,
                }
            ]
        )
        return bool(responses and responses[0].get("status") == "ok")

    def add_edge(
        self,
        src: int,
        dst: int,
        relation: str = "RELATED_TO",
        weight: float = 1.0,
    ) -> bool:
        responses = self._send(
            [
                {
                    "op": "edge",
                    "src": src,
                    "dst": dst,
                    "relation": relation,
                    "weight": weight,
                }
            ]
        )
        return bool(responses and responses[0].get("status") == "ok")

    def add_nodes_batch(self, nodes: list[dict]) -> int:
        """Add multiple nodes at once. Returns count of successful writes."""
        ops = [
            {
                "op": "node",
                "id": n["id"],
                "label": n["label"],
                "content": n["content"],
                "confidence": n.get("confidence", 0.9),
            }
            for n in nodes
        ]
        responses = self._send(ops)
        return sum(1 for r in responses if r.get("status") == "ok")

    def count(self) -> int:
        responses = self._send([{"op": "count"}])
        if responses and responses[0].get("status") == "ok":
            return responses[0].get("count", 0)
        return 0
