#!/usr/bin/env python3
"""
将现有日志迁移为统一结构化格式（JSONL），并输出到全新目录，不覆盖原日志。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = "/mnt/e/Projects/eriq-eval"
DEFAULT_SRC_DIR = f"{PROJECT_ROOT}/logs"
DEFAULT_DST_ROOT = f"{PROJECT_ROOT}/logs_v2"

STD_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+([A-Z]+)\s+\|")
ERR_RE = re.compile(r"(traceback|exception|error|oom|out of memory|\[fail\]|失败)", re.IGNORECASE)
WARN_RE = re.compile(r"(warn|warning|警告)", re.IGNORECASE)


def infer_level(text: str) -> str:
    if ERR_RE.search(text):
        return "ERROR"
    if WARN_RE.search(text):
        return "WARN"
    return "INFO"


def normalize_line(raw: str, source_file: str, source_line: int, migrated_at: str) -> Dict[str, Any]:
    line = raw.rstrip("\n")
    module = Path(source_file).stem

    # 1) 兼容已有 jsonl
    if line.startswith("{") and line.endswith("}"):
        try:
            payload = json.loads(line)
            message = payload.get("message") or payload.get("event") or payload.get("type") or line
            return {
                "timestamp": payload.get("ts") or payload.get("timestamp"),
                "level": str(payload.get("level", infer_level(message))).upper(),
                "pid": payload.get("pid"),
                "module": payload.get("module") or module,
                "message": message,
                "source_file": source_file,
                "source_line": source_line,
                "migrated_at": migrated_at,
                "raw": payload,
            }
        except json.JSONDecodeError:
            pass

    # 2) 兼容标准文本日志头
    ts_match = STD_TS_RE.match(line)
    if ts_match:
        ts, level = ts_match.group(1), ts_match.group(2)
        return {
            "timestamp": ts,
            "level": level,
            "pid": None,
            "module": module,
            "message": line,
            "source_file": source_file,
            "source_line": source_line,
            "migrated_at": migrated_at,
            "raw": None,
        }

    # 3) 普通文本
    return {
        "timestamp": None,
        "level": infer_level(line),
        "pid": None,
        "module": module,
        "message": line,
        "source_file": source_file,
        "source_line": source_line,
        "migrated_at": migrated_at,
        "raw": None,
    }


def walk_log_files(src_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".log", ".jsonl"}:
            out.append(p)
    return sorted(out)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="迁移历史日志到统一格式（JSONL）")
    parser.add_argument("--src-dir", default=DEFAULT_SRC_DIR, help="源日志目录")
    parser.add_argument("--dst-root", default=DEFAULT_DST_ROOT, help="目标根目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    dst_root = Path(args.dst_root).resolve()

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    migrated_at = datetime.now().isoformat(timespec="seconds")
    dst_dir = dst_root / f"migrated_{ts_tag}"
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = walk_log_files(src_dir)
    if not files:
        print(f"[INFO] 未发现可迁移日志: {src_dir}")
        return 0

    manifest_rows = []
    total_rows = 0
    total_bytes = 0
    total_level = Counter()

    for src in files:
        rel = src.relative_to(src_dir)
        dst = (dst_dir / rel).with_suffix(rel.suffix + ".jsonl")
        text = src.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        rows = [normalize_line(line, str(src), i + 1, migrated_at) for i, line in enumerate(lines)]
        written = write_jsonl(dst, rows)

        level_counter = Counter(r["level"] for r in rows)
        size = src.stat().st_size
        total_rows += written
        total_bytes += size
        total_level.update(level_counter)
        manifest_rows.append(
            {
                "source_file": str(src),
                "target_file": str(dst),
                "source_size_bytes": size,
                "line_count": len(lines),
                "written_count": written,
                "level_counter": dict(level_counter),
            }
        )
        print(f"[OK] {src} -> {dst} ({written} lines)")

    manifest_json = dst_dir / "migration_manifest.json"
    manifest_md = dst_dir / "migration_manifest.md"
    manifest_json.write_text(
        json.dumps(
            {
                "migrated_at": migrated_at,
                "source_dir": str(src_dir),
                "target_dir": str(dst_dir),
                "file_count": len(files),
                "total_lines": total_rows,
                "total_source_bytes": total_bytes,
                "total_level_counter": dict(total_level),
                "files": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# 日志迁移清单",
        "",
        f"- 迁移时间: `{migrated_at}`",
        f"- 源目录: `{src_dir}`",
        f"- 目标目录: `{dst_dir}`",
        f"- 文件数: `{len(files)}`",
        f"- 总行数: `{total_rows}`",
        f"- 源日志总大小: `{total_bytes}` bytes",
        f"- 级别汇总: `{dict(total_level)}`",
        "",
        "| 源文件 | 新文件 | 行数 | 原始大小(bytes) |",
        "|---|---|---:|---:|",
    ]
    for row in manifest_rows:
        md_lines.append(
            f"| {row['source_file']} | {row['target_file']} | {row['line_count']} | {row['source_size_bytes']} |"
        )
    manifest_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] 迁移清单(JSON): {manifest_json}")
    print(f"[OK] 迁移清单(MD): {manifest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
