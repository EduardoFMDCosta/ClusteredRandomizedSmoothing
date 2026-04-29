from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    setting = item.get("setting", {})
    return (
        setting.get("alpha"),
        setting.get("N"),
        setting.get("max_num_clusters"),
        setting.get("coverage"),
        setting.get("sigma"),
        setting.get("r"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded analysis_trajflow JSON outputs into one file.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="experiments/data/analysis_parallel",
        help="Directory containing results_shard_*.json files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results_shard_*_of_*.json",
        help="Glob pattern to match shard files in input-dir.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results.json",
        help="Output merged JSON filename (written inside input-dir unless absolute path).",
    )
    parser.add_argument(
        "--expected-num-shards",
        type=int,
        default=None,
        help="Optional strict check on expected shard file count.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    shard_files = sorted(glob.glob(str(input_dir / args.pattern)))

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {input_dir} with pattern {args.pattern}")

    if args.expected_num_shards is not None and len(shard_files) != args.expected_num_shards:
        raise ValueError(
            f"Expected {args.expected_num_shards} shard files, found {len(shard_files)}"
        )

    merged: list[dict[str, Any]] = []
    for path in shard_files:
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Shard file is not a list: {path}")
        merged.extend(payload)

    merged.sort(key=sort_key)

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = input_dir / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(shard_files)} shards into {output_path}")
    print(f"Total settings in merged file: {len(merged)}")


if __name__ == "__main__":
    main()
