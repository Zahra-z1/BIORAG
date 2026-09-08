import argparse
from src.create_index import build_index


def main():
    parser = argparse.ArgumentParser(description="Build BIO RAG knowledge-base indexes.")
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Build metadata only; do not download/load the embedding model or create FAISS.",
    )
    args = parser.parse_args()
    build_index(build_semantic=not args.lexical_only)


if __name__ == "__main__":
    main()
