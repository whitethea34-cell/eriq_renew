#!/usr/bin/env python3
"""
生成 ERIQ 模型系统性排查报告（基于 run_list/state/log 的离线分析）

说明:
- 不执行模型推理，仅分析已有配置、状态和日志
- 输出:
  1) Markdown 报告
  2) JSON 结构化报�?"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import os
import re
from typing import Any, Dict, List, Optional

import yaml


@dataclasses.dataclass
class ErrorEvent:
    category: str
    message: str
    line_no: int
    source_log: str
    task_id: Optional[str]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def task_id_of(item: Dict[str, Any]) -> str:
    return f"{item.get('series')}|{item.get('size')}|{item.get('quant', 'fp16')}"


def normalize_error_category(msg: str) -> str:
    s = (msg or "").lower()
    if "wrong pixel_values size" in s:
        return "data_tensor_shape_mismatch"
    if "out of memory" in s or "cudacachingallocator" in s:
        return "cuda_oom"
    if "cuda error: unknown error" in s:
        return "cuda_unknown_error"
    if "no valid images loaded" in s:
        return "image_path_or_data_error"
    if "has no attribute 'cb'" in s or "all_tied_weights_keys" in s:
        return "quant_or_loader_compatibility"
    if "snapshot_download" in s or "connection" in s or "timeout" in s:
        return "model_download_or_network"
    return "other_runtime_error"


def parse_log_events(log_path: str) -> List[ErrorEvent]:
    if not os.path.exists(log_path):
        return []

    events: List[ErrorEvent] = []
    current_task: Optional[str] = None
    task_start_re = re.compile(r"\[task\] run start: ([^ ]+)")
    err_re = re.compile(r"\[错误\].*?:\s*(.*)$")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            m = task_start_re.search(line)
            if m:
                current_task = m.group(1)

            m = err_re.search(line)
            if m:
                msg = m.group(1).strip()
                events.append(
                    ErrorEvent(
                        category=normalize_error_category(msg),
                        message=msg,
                        line_no=idx,
                        source_log=log_path,
                        task_id=current_task,
                    )
                )
                continue

            # 捕获�?[错误] 形式的关键异�?            if "CUDA error: unknown error" in line:
                events.append(
                    ErrorEvent(
                        category="cuda_unknown_error",
                        message="CUDA error: unknown error",
                        line_no=idx,
                        source_log=log_path,
                        task_id=current_task,
                    )
                )
            elif "CUDA out of memory" in line or "driver error: out of memory" in line:
                events.append(
                    ErrorEvent(
                        category="cuda_oom",
                        message=line.strip(),
                        line_no=idx,
                        source_log=log_path,
                        task_id=current_task,
                    )
                )
    return events


def infer_priority(category: str) -> str:
    if category in {"data_tensor_shape_mismatch", "quant_or_loader_compatibility"}:
        return "P0"
    if category in {"cuda_oom", "model_download_or_network"}:
        return "P1"
    if category in {"cuda_unknown_error"}:
        return "P2"
    return "P3"


def build_model_checklist(
    runs: List[Dict[str, Any]],
    state: Dict[str, Any],
    events: List[ErrorEvent],
) -> List[Dict[str, Any]]:
    by_task_events: Dict[str, List[ErrorEvent]] = collections.defaultdict(list)
    for e in events:
        if e.task_id:
            by_task_events[e.task_id].append(e)

    checklist: List[Dict[str, Any]] = []
    for item in runs:
        tid = task_id_of(item)
        task_state = state.get("tasks", {}).get(tid, {})
        status = task_state.get("status", "unknown")
        errs = by_task_events.get(tid, [])
        categories = sorted({e.category for e in errs})
        has_shape = "data_tensor_shape_mismatch" in categories
        has_oom = "cuda_oom" in categories

        if status == "completed":
            data_fmt = "PASS"
            tensor_flow = "PASS"
            load_init = "PASS"
            infer_ex = "PASS"
            overall = "PASS"
        else:
            data_fmt = "FAIL" if has_shape else "PENDING"
            tensor_flow = "FAIL" if has_shape else ("RISK" if has_oom else "PENDING")
            load_init = "PENDING"
            infer_ex = "FAIL" if (has_shape or has_oom) else "PENDING"
            overall = "FAIL" if (has_shape or has_oom) else "PENDING"

        if status in {"in_progress"} and overall == "PENDING":
            overall = "RUNNING"

        checklist.append(
            {
                "task_id": tid,
                "series": item.get("series"),
                "size": str(item.get("size")),
                "quant": item.get("quant", "fp16"),
                "env": item.get("env"),
                "config": item.get("config"),
                "state_status": status,
                "check_data_format": data_fmt,
                "check_tensor_flow": tensor_flow,
                "check_load_init": load_init,
                "check_inference_exception": infer_ex,
                "overall": overall,
                "error_categories": categories,
                "last_reason": task_state.get("last_reason"),
            }
        )
    return checklist


def to_markdown(
    report: Dict[str, Any],
    checklist: List[Dict[str, Any]],
    output_md: str,
) -> None:
    lines: List[str] = []
    lines.append("# ERIQ 模型系统性排查报告（离线分析�?)
    lines.append("")
    lines.append(f"- 生成时间: `{report['generated_at']}`")
    lines.append(f"- 项目根目�? `{report['project_root']}`")
    lines.append(f"- 模型总数: `{report['model_count']}`")
    lines.append(f"- 系列分布: `{report['series_distribution']}`")
    lines.append(f"- 状态分�? `{report['state_distribution']}`")
    lines.append("")
    lines.append("## 1) 问题分布统计")
    lines.append("")
    lines.append("| 根因类别 | 数量 | 优先�?|")
    lines.append("|---|---:|---|")
    for cat, cnt in report["error_distribution"]:
        lines.append(f"| `{cat}` | {cnt} | `{infer_priority(cat)}` |")
    lines.append("")
    lines.append("## 2) 根因分类与修复优先级")
    lines.append("")
    lines.append("- `P0` 数据/张量结构问题: 典型�?`wrong pixel_values size`，会导致整模型整子任务全失败�?)
    lines.append("- `P1` 资源与下载问�? `cuda_oom`、模型拉取不稳定，影响稳定性与覆盖率�?)
    lines.append("- `P2` CUDA 异步未知错误: 常由上游 CUDA 异常污染触发�?)
    lines.append("- `P3` 其他运行时错�? 需按具体栈定位�?)
    lines.append("")
    lines.append("## 3) 栈样例（每类首条�?)
    lines.append("")
    for cat, sample in report["error_samples"].items():
        lines.append(f"- `{cat}`: `{sample}`")
    lines.append("")
    lines.append("## 4) 每模型检查清�?)
    lines.append("")
    lines.append("| 模型(task_id) | 数据格式 | 张量流程 | 加载初始�?| 推理异常 | 综合状�?| 现状 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in checklist:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | `{}` |".format(
                row["task_id"],
                row["check_data_format"],
                row["check_tensor_flow"],
                row["check_load_init"],
                row["check_inference_exception"],
                row["overall"],
                row["state_status"],
            )
        )
    lines.append("")
    lines.append("## 5) 复现步骤（统一�?)
    lines.append("")
    lines.append("```bash")
    lines.append("cd /mnt/e/Projects/eriq-eval")
    lines.append("python stepbystep/engine/run_from_config.py \\")
    lines.append("  --config /mnt/e/Projects/eriq-eval/stepbystep/configs/fp16_budget/<model>.yaml \\")
    lines.append("  --subset-file QA_RELATIVE_POS_GROUNDING.json \\")
    lines.append("  --limit 1 \\")
    lines.append("  --output /mnt/e/Projects/eriq-eval/results/_diag_<model>.json")
    lines.append("```")
    lines.append("")
    lines.append("## 6) 结论")
    lines.append("")
    lines.append("- 现阶段首要问题是 `data_tensor_shape_mismatch`，优先完成所�?InternVL 路径验证�?)
    lines.append("- 完成结构性修复后，再集中处理 `cuda_oom` �?`cuda_unknown_error`�?)
    lines.append("- 建议使用 `stepbystep/diagnostics/run_system_diagnostic.sh` 做批量可重复检查�?)
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成系统性排查报告（离线�?)
    parser.add_argument(
        "--run-list",
        default="/mnt/e/Projects/eriq-eval/stepbystep/configs/fp16_budget/run_list_fp16_budget.yaml",
    )
    parser.add_argument(
        "--state-file",
        default="/mnt/e/Projects/eriq-eval/runtime/smart_retry/retry_state.json",
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            "/mnt/e/Projects/eriq-eval/logs/smart_retry/smart_retry_daemon.log",
            "/mnt/e/Projects/eriq-eval/logs/retry_failed_models_fp16_20b.log",
        ],
    )
    parser.add_argument(
        "--out-md",
        default="/mnt/e/Projects/eriq-eval/stepbystep/diagnostics/system_diagnostic_report.md",
    )
    parser.add_argument(
        "--out-json",
        default="/mnt/e/Projects/eriq-eval/stepbystep/diagnostics/system_diagnostic_report.json",
    )
    args = parser.parse_args()

    run_data = load_yaml(args.run_list)
    runs = run_data.get("runs", [])
    state = load_json(args.state_file) if os.path.exists(args.state_file) else {}
    events: List[ErrorEvent] = []
    for log in args.logs:
        events.extend(parse_log_events(log))

    err_counter = collections.Counter([e.category for e in events])
    err_samples: Dict[str, str] = {}
    for e in events:
        if e.category not in err_samples:
            err_samples[e.category] = f"{e.message} @ {e.source_log}:{e.line_no}"

    series_counter = collections.Counter([r.get("series") for r in runs])
    state_counter = collections.Counter()
    for r in runs:
        tid = task_id_of(r)
        st = state.get("tasks", {}).get(tid, {}).get("status", "unknown")
        state_counter[st] += 1

    checklist = build_model_checklist(runs, state, events)
    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "project_root": "/mnt/e/Projects/eriq-eval",
        "model_count": len(runs),
        "series_distribution": dict(series_counter),
        "state_distribution": dict(state_counter),
        "error_distribution": err_counter.most_common(),
        "error_samples": err_samples,
        "model_checklist": checklist,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    to_markdown(report, checklist, args.out_md)

    print(f"[ok] wrote: {args.out_json}")
    print(f"[ok] wrote: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
