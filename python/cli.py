#!/usr/bin/env python3
"""
Socratic Engine CLI
Runs the debate_runner binary and pretty-prints the JSON stream.
"""

import subprocess
import json
import sys
import os
import argparse
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BINARY = PROJECT_ROOT / "build/cpp/dispatcher/debate_runner"
LLAMA_LIBS = Path.home() / "llama.cpp/build/bin"

ROLE_COLORS = {
    "Proposer": "\033[94m",  # blue
    "Critic": "\033[91m",  # red
    "Retriever": "\033[92m",  # green
    "Synthesizer": "\033[93m",  # yellow
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def strip_artifacts(text: str) -> str:
    """Remove Phi-3 chat tokens from output."""
    for token in ["<|end|>", "<|assistant|>", "<|user|>", "<|system|>", "<s>", "</s>"]:
        text = text.replace(token, "")
    return text.strip()


def run_debate(query: str, model: str = None, tokens: int = 150) -> int:
    if not BINARY.exists():
        print(f"Error: debate_runner not found at {BINARY}")
        print("Run: cmake --build build --target debate_runner")
        return 1

    cmd = [
        str(BINARY),
        "--query",
        query,
        "--tokens",
        str(tokens),
    ]
    if model:
        cmd += ["--model", model]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_LIBS)

    print(f"\n{BOLD}Socratic Engine{RESET}")
    print(f"{DIM}Query: {query}{RESET}\n")
    print("─" * 60)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # silence llama.cpp logs
            text=True,
            env=env,
            bufsize=1,  # line-buffered — stream as rounds complete
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("event")

            if etype == "start":
                nodes = event.get("kg_nodes", 0)
                print(f"{DIM}Knowledge graph: {nodes} node(s) loaded{RESET}\n")

            elif etype == "round":
                role = event.get("role", "Agent")
                round_ = event.get("round", 0)
                output = strip_artifacts(event.get("output", ""))
                color = ROLE_COLORS.get(role, "")

                print(f"{color}{BOLD}[Round {round_}] {role}{RESET}")
                print(output)
                print()

            elif etype == "done":
                elapsed = event.get("elapsed", 0)
                success = event.get("success", False)
                print("─" * 60)
                status = "✓ Complete" if success else "✗ Failed"
                print(f"{DIM}{status} in {elapsed:.1f}s{RESET}\n")

            elif etype == "error":
                msg = event.get("message", "unknown error")
                print(f"\033[91mError: {msg}{RESET}", file=sys.stderr)
                return 1

        proc.wait()
        return proc.returncode

    except KeyboardInterrupt:
        proc.terminate()
        print(f"\n{DIM}Interrupted.{RESET}")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Socratic Engine — multi-agent LLM debate CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "What causes lightning?"
  python cli.py "Explain black holes" --tokens 200
  python cli.py "How does photosynthesis work?" --model ~/models/phi3-mini-q4.gguf
        """,
    )
    parser.add_argument("query", nargs="?", help="Question to debate")
    parser.add_argument("--model", help="Path to GGUF model file")
    parser.add_argument(
        "--tokens", type=int, default=150, help="Max tokens per agent (default: 150)"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode — ask multiple questions",
    )

    args = parser.parse_args()

    if args.interactive:
        print(f"{BOLD}Socratic Engine — Interactive Mode{RESET}")
        print(f"{DIM}Type your question and press Enter. Ctrl+C to exit.{RESET}\n")
        while True:
            try:
                query = input("Question: ").strip()
                if not query:
                    continue
                run_debate(query, args.model, args.tokens)
            except (KeyboardInterrupt, EOFError):
                print(f"\n{DIM}Goodbye.{RESET}")
                break
        return 0

    if not args.query:
        parser.print_help()
        return 1

    return run_debate(args.query, args.model, args.tokens)


if __name__ == "__main__":
    sys.exit(main())
