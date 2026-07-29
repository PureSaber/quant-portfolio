from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quant_portfolio.allocator import allocate, load_config


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    snap = allocate(cfg)
    payload = {
        "as_of": snap.as_of,
        "total_nav": snap.total_nav,
        "books": snap.books,
        "combined_weights": snap.combined_weights,
    }
    out = Path(args.out) if args.out else None
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant-portfolio")
    sub = p.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Combine strategy books into portfolio view")
    status.add_argument("--config", required=True)
    status.add_argument("--out", default="")
    status.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
