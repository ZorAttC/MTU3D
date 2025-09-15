#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取导航结果 JSON，计算各 split 的 SR、SPL 以及条目数，可选按 scan_id 聚合"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="output_dirs/ovon-full-finetune-num-1.json",
        help="输入 JSON 文件路径（默认：output_dirs/ovon-full-finetune-num-1.json）",
    )
    parser.add_argument(
        "--by-scan",
        action="store_true",
        help="是否按 scan_id（房间）聚合统计 SR、SPL 与条目数",
    )
    return parser.parse_args()


def to_bool_sr(value: Any) -> int:
    # 有些结果里 sr 可能是 True/False，也可能是 0/1
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return 1 if int(value) != 0 else 0
    except Exception:
        return 0


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def compute_stats(records: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    n = len(records)
    if n == 0:
        return 0, 0.0, 0.0
    sr_sum = 0
    spl_sum = 0.0
    for r in records:
        sr_sum += to_bool_sr(r.get("sr", 0))
        spl_sum += to_float(r.get("spl", 0.0))
    sr_mean = sr_sum / n
    spl_mean = spl_sum / n
    return n, sr_mean, spl_mean


def compute_stats_by_scan(records: List[Dict[str, Any]]) -> Dict[str, Tuple[int, float, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        scan_id = r.get("scan_id", "UNKNOWN")
        buckets[scan_id].append(r)
    per_scan: Dict[str, Tuple[int, float, float]] = {}
    for scan_id, recs in buckets.items():
        per_scan[scan_id] = compute_stats(recs)
    return per_scan


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 仅处理值为 list 的键（如 val_seen、val_unseen 等）
    splits = {k: v for k, v in data.items() if isinstance(v, list)}
    if not splits:
        print("未在 JSON 中找到可用 split 列表。")
        return

    print("总体统计：")
    for split_name, records in splits.items():
        n, sr, spl = compute_stats(records)
        print(f"- {split_name}: count={n}, SR={sr:.4f}, SPL={spl:.4f}")

    if args.by_scan:
        print("\n按 scan_id（房间）统计：")
        for split_name, records in splits.items():
            print(f"[{split_name}]")
            per_scan = compute_stats_by_scan(records)
            # 按 scan_id 排序，方便查看
            for scan_id in sorted(per_scan.keys()):
                n, sr, spl = per_scan[scan_id]
                print(f"  scan_id={scan_id}: count={n}, SR={sr:.4f}, SPL={spl:.4f}")


if __name__ == "__main__":
    main()
