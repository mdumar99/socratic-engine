#!/usr/bin/env python3
"""
KG Builder CLI — populate the knowledge graph from text sources.

Usage:
  python kg_builder/cli.py add --text "Photosynthesis is..." --kg /tmp/mykg
  python kg_builder/cli.py add --file docs/biology.txt --kg /tmp/mykg
  python kg_builder/cli.py add --url https://en.wikipedia.org/wiki/Lightning --kg /tmp/mykg
  python kg_builder/cli.py count --kg /tmp/mykg
  python kg_builder/cli.py edge --src 1 --dst 2 --relation CAUSES --kg /tmp/mykg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.kg_builder.builder import KGBuilder
from python.kg_builder.sources import (
    nodes_from_file,
    nodes_from_url,
    nodes_from_string,
)

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def cmd_add(args):
    kg = KGBuilder(args.kg)

    if args.text:
        nodes = nodes_from_string(
            args.text, label=args.label or "text", base_id=args.base_id
        )
    elif args.file:
        print(f"{DIM}Loading {args.file}...{RESET}")
        nodes = nodes_from_file(args.file, base_id=args.base_id)
    elif args.url:
        print(f"{DIM}Fetching {args.url}...{RESET}")
        nodes = nodes_from_url(args.url, base_id=args.base_id)
    else:
        print(f"{RED}Error: provide --text, --file, or --url{RESET}")
        return 1

    print(f"{DIM}Writing {len(nodes)} nodes to {args.kg}...{RESET}")
    written = kg.add_nodes_batch(nodes)
    total = kg.count()

    print(
        f"{GREEN}✓ {written}/{len(nodes)} nodes written "
        f"· KG total: {total} nodes{RESET}"
    )
    return 0


def cmd_count(args):
    kg = KGBuilder(args.kg)
    count = kg.count()
    print(f"KG at {args.kg}: {BOLD}{count}{RESET} nodes")
    return 0


def cmd_edge(args):
    kg = KGBuilder(args.kg)
    ok = kg.add_edge(args.src, args.dst, args.relation, args.weight)
    if ok:
        print(f"{GREEN}✓ Edge {args.src} --{args.relation}--> {args.dst}{RESET}")
    else:
        print(f"{RED}✗ Failed to write edge{RESET}")
    return 0 if ok else 1


def cmd_node(args):
    kg = KGBuilder(args.kg)
    ok = kg.add_node(args.id, args.label, args.content, args.confidence)
    if ok:
        print(f"{GREEN}✓ Node {args.id}: {args.label}{RESET}")
    else:
        print(f"{RED}✗ Failed to write node{RESET}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Socratic Engine — KG Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add text content to the KG")
    p_add.add_argument("--kg", default="/tmp/se_runtime_kg")
    p_add.add_argument("--text", help="Plain text string to add")
    p_add.add_argument("--file", help="Path to text/markdown file")
    p_add.add_argument("--url", help="URL to fetch and add")
    p_add.add_argument("--label", default="text", help="Label prefix for nodes")
    p_add.add_argument(
        "--base-id",
        type=int,
        default=1000,
        dest="base_id",
        help="Starting node ID (default: 1000)",
    )

    # count
    p_count = sub.add_parser("count", help="Count nodes in the KG")
    p_count.add_argument("--kg", default="/tmp/se_runtime_kg")

    # edge
    p_edge = sub.add_parser("edge", help="Add an edge between two nodes")
    p_edge.add_argument("--kg", default="/tmp/se_runtime_kg")
    p_edge.add_argument("--src", type=int, required=True)
    p_edge.add_argument("--dst", type=int, required=True)
    p_edge.add_argument(
        "--relation",
        default="RELATED_TO",
        choices=[
            "IS_A",
            "HAS_PROPERTY",
            "CAUSES",
            "CONTRADICTS",
            "SUPPORTS",
            "PART_OF",
            "RELATED_TO",
        ],
    )
    p_edge.add_argument("--weight", type=float, default=1.0)

    # node
    p_node = sub.add_parser("node", help="Add a single node manually")
    p_node.add_argument("--kg", default="/tmp/se_runtime_kg")
    p_node.add_argument("--id", type=int, required=True)
    p_node.add_argument("--label", required=True)
    p_node.add_argument("--content", default="")
    p_node.add_argument("--confidence", type=float, default=0.9)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "add": cmd_add,
        "count": cmd_count,
        "edge": cmd_edge,
        "node": cmd_node,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
