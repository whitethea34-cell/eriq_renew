#!/usr/bin/env python3
"""
�?QA_* 子任务分段运行指定模型配置：
- 每个子任务单独启动一�?run_from_config.py（模型会重新加载/释放�?- 最终自动合并为一个输�?JSON，便�?eval_hf.py �?summary 汇�?"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import List


def list_subset_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "QA_*.json")))
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到 QA_*.json")
    return files


def main():
    parser = argparse.ArgumentParser(description="按子任务分段运行 run_from_config.py 并合并结�?)
    parser.add_argument("--config", required=True, help="模型配置文件")
    parser.add_argument("--data-dir", default="/mnt/e/Projects/eriq-eval/data", help="ERIQ 数据目录")
    parser.add_argument("--output", required=True, help="最终合并输�?JSON")
    parser.add_argument("--limit", type=int, default=None, help="每个子任务仅取前 N �?)
    args = parser.parse_args()

    config = os.path.abspath(os.path.expanduser(args.config))
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    output = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)

    subset_files = list_subset_files(data_dir)
    merged = []

    with tempfile.TemporaryDirectory(prefix="eriq_subset_") as tmpdir:
        for subset_path in subset_files:
            subset_name = os.path.basename(subset_path).replace(".json", "")
            tmp_out = os.path.join(tmpdir, f"{subset_name}.json")
            cmd = [
                sys.executable,
                "/mnt/e/Projects/eriq-eval/stepbystep/engine/run_from_config.py",
                "--config", config,
                "--data-dir", data_dir,
                "--subset-file", os.path.basename(subset_path),
                "--output", tmp_out,
            ]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]

            print(f"[分段运行] {subset_name}")
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"[失败] 子任务运行失�? {subset_name} (exit={rc})")
                # 加载级错误通常会在首个子任务立即失败，快速失败可避免 15 个子任务重复报错�?                fast_fail = os.getenv("ERIQ_FAST_FAIL_BOOTSTRAP", "1").strip() == "1"
                if fast_fail and len(merged) == 0:
                    print("[快速失败] 首个子任务失败且暂无有效输出，提前终止本模型分段运行�?)
                    break
                continue
            if not os.path.exists(tmp_out):
                print(f"[失败] 未生成输出文�? {tmp_out}")
                continue
            with open(tmp_out, "r", encoding="utf-8") as f:
                rows = json.load(f)
            merged.extend(rows)
            print(f"[完成] {subset_name} -> {len(rows)} �?)
            cooldown = os.getenv("ERIQ_SUBSET_COOLDOWN_SEC", "").strip()
            try:
                cooldown_sec = int(cooldown) if cooldown else 0
            except Exception:
                cooldown_sec = 0
            if cooldown_sec > 0:
                print(f"[冷却] 子任务间等待 {cooldown_sec}s")
                time.sleep(cooldown_sec)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[完成] 合并输出: {output} ({len(merged)} �?")


if __name__ == "__main__":
    main()
