"""Command-line interface for the Gemini documentation tutor."""

import argparse
from pathlib import Path

from gemini_agent.qa import answer_question, build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini-powered documentation tutor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("build-index", help="Create an embedding index")
    index_parser.add_argument("source", type=Path, help="Directory containing documentation files")
    index_parser.add_argument("output", type=Path, help="Where to store the index JSON")
    index_parser.add_argument("--chunk-size", type=int, default=800, help="Characters per chunk")
    index_parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index")
    ask_parser.add_argument("index", type=Path, help="Path to the saved index JSON")
    ask_parser.add_argument("question", type=str, help="Question to ask the agent")
    ask_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build-index":
        index_path = build_index(args.source, args.output, args.chunk_size, args.overlap)
        print(f"Index saved to {index_path}")
    elif args.command == "ask":
        answer = answer_question(args.question, args.index, top_k=args.top_k)
        print("\n=== Agent Response ===\n")
        print(answer)


if __name__ == "__main__":
    main()
