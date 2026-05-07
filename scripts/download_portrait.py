#!/usr/bin/env python3
"""Print acquisition instructions for portrait datasets.

The portrait datasets used in Table 2 (VFHQ, CelebV-HQ) are gated and
cannot be auto-downloaded.  This script writes a short README to the
requested directory describing how to obtain them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emg.data.download import PORTRAIT_DATASET_INFO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("./portrait_data_info.md"))
    p.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.json:
        args.output.with_suffix(".json").write_text(
            json.dumps(PORTRAIT_DATASET_INFO, indent=2)
        )
        print(f"wrote {args.output.with_suffix('.json')}")
        return 0

    lines: list[str] = ["# Portrait dataset acquisition\n"]
    for name, info in PORTRAIT_DATASET_INFO.items():
        lines.append(f"## {name}\n")
        lines.append(f"- Description: {info.get('description', 'n/a')}")
        lines.append(f"- URL: {info.get('url', 'n/a')}")
        lines.append(f"- License: {info.get('license', 'n/a')}")
        lines.append(f"- How to obtain: {info.get('how_to_obtain', 'n/a')}")
        lines.append("")
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
