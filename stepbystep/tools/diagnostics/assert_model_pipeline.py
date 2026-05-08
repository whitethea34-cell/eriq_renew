#!/usr/bin/env python3
"""
模型管线断言工具（可选执行）

用�?
1) 静态断言（默认）: 配置字段、输入输出协议、关键参�?2) 动态断言�?-execute�? �?limit=1，检查输�?JSON 结构和预测字�?"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List

import yaml


REQUIRED_ROOT_KEYS = ["model_name", "model_id", "loader", "inference"]
REQUIRED_LOADER_KEYS = ["class", "processor_class", "trust_remote_code", "torch_dtype"]
REQUIRED_INFER_KEYS = ["max_new_tokens", "do_sample", "system_prompt"]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def assert_static(cfg: Dict[str, Any], config_path: str) -> List[str]:
    errs: List[str] = []
    for k in REQUIRED_ROOT_KEYS:
        if k not in cfg:
            errs.append(f"missing root key: {k}")
    loader = cfg.get("loader", {})
    for k in REQUIRED_LOADER_KEYS:
        if k not in loader:
            errs.append(f"missing loader key: {k}")
    infer = cfg.get("inference", {})
    for k in REQUIRED_INFER_KEYS:
        if k not in infer:
            errs.append(f"missing inference key: {k}")

    max_new_tokens = infer.get("max_new_tokens")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        errs.append(f"invalid max_new_tokens: {max_new_tokens}")
    if infer.get("do_sample") is not False:
        errs.append("do_sample should be false for deterministic eval")

    model_name = str(cfg.get("model_name", "")).lower()
    if "internvl" in model_name:
        # InternVL 重点约束: 需�?remote code + 可处理图像的 processor
        if not bool(loader.get("trust_remote_code")):
            errs.append("internvl requires trust_remote_code=true")
        if "tokenizer" not in str(loader.get("processor_class", "")).lower():
            errs.append("internvl processor_class expected tokenizer-like remote processor")

    if not os.path.isabs(config_path):
        errs.append("config path should be absolute in test runner")
    return errs


def assert_dynamic(output_path: str) -> List[str]:
    errs: List[str] = []
    if not os.path.exists(output_path):
        return [f"output not found: {output_path}"]
    with open(output_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return ["output json is not a list"]
    if not rows:
        return ["output list is empty"]

    required_fields = ["id", "prediction", "gt_answer", "raw_output"]
    for field in required_fields:
        if field not in rows[0]:
            errs.append(f"missing output field: {field}")
    pred = rows[0].get("prediction", "")
    if pred not in {"A", "B", "C", "D", "E", "F", "Yes", "No", ""}:
        errs.append(f"prediction format invalid: {pred}")
    return errs


def run_one_sample(config: str, subset: str, output: str, data_dir: str) -> int:
    cmd = (
        "python /mnt/e/Projects/eriq-eval/stepbystep/engine/run_from_config.py "
        f"--config {shlex.quote(config)} --subset-file {shlex.quote(subset)} "
        f"--limit 1 --data-dir {shlex.quote(data_dir)} --output {shlex.quote(output)}"
    )
    print(f"[exec] {cmd}")
    return subprocess.call(["bash", "-lc", cmd])


def main() -> int:
    parser = argparse.ArgumentParser(description="ERIQ 模型管线断言")
    parser.add_argument("--config", required=True, help="模型配置 YAML（绝对路径）")
    parser.add_argument("--subset", default="QA_RELATIVE_POS_GROUNDING.json")
    parser.add_argument("--data-dir", default="/mnt/e/Projects/eriq-eval/data")
    parser.add_argument("--output", default="/mnt/e/Projects/eriq-eval/results/_diag_assert_output.json")
    parser.add_argument("--execute", action="store_true", help="执行 limit=1 动态验�?)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    static_errs = assert_static(cfg, args.config)
    if static_errs:
        print("[static] FAIL")
        for e in static_errs:
            print(f" - {e}")
        return 2
    print("[static] PASS")

    if not args.execute:
        print("[dynamic] SKIPPED (use --execute)")
        return 0

    rc = run_one_sample(args.config, args.subset, args.output, args.data_dir)
    if rc != 0:
        print(f"[dynamic] FAIL: run_from_config exit={rc}")
        return rc

    dyn_errs = assert_dynamic(args.output)
    if dyn_errs:
        print("[dynamic] FAIL")
        for e in dyn_errs:
            print(f" - {e}")
        return 3
    print("[dynamic] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
