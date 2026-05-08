#!/usr/bin/env python3
"""
汇总冒烟结果:
- 平均时延
- P50 / P95
- 吞吐
- 有效预测率
- 错误计数
- 通过判定
"""

import os
import json
import argparse
import statistics
import re
import yaml

VALID_SET = {"A", "B", "C", "D", "E", "F", "Yes", "No"}


def percentile(values, p):
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, int(len(arr) * p))
    return arr[idx]


def summarize_one(path):
    if not os.path.exists(path):
        return {
            "exists": False,
            "total": 0,
            "valid_rate": 0.0,
            "error_count": 0,
            "avg_latency": 0.0,
            "p50_latency": 0.0,
            "p95_latency": 0.0,
            "throughput": 0.0,
            "pass": False,
            "reason": "missing_file",
        }

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    total = len(rows)
    valid = 0
    errors = 0
    lats = []
    for r in rows:
        pred = str(r.get("prediction", "")).strip()
        if pred in VALID_SET:
            valid += 1
        if str(r.get("raw_output", "")).startswith("ERROR:"):
            errors += 1
        lat = r.get("latency_sec")
        if isinstance(lat, (float, int)):
            lats.append(float(lat))

    valid_rate = (valid / total) if total else 0.0
    avg_latency = statistics.mean(lats) if lats else 0.0
    p50 = statistics.median(lats) if lats else 0.0
    p95 = percentile(lats, 0.95) if lats else 0.0
    throughput = (len(lats) / sum(lats)) if lats and sum(lats) > 0 else 0.0

    # 通过规则：只要结果文件存在、且没有 ERROR 记录，就视为通过。
    passed = (
        total > 0 and
        errors == 0
    )
    reason = "ok" if passed else "failed_rule"

    return {
        "exists": True,
        "total": total,
        "valid_rate": valid_rate,
        "error_count": errors,
        "avg_latency": avg_latency,
        "p50_latency": p50,
        "p95_latency": p95,
        "throughput": throughput,
        "pass": passed,
        "reason": reason,
    }


def parse_score_accuracy(score_path):
    """
    从 *_score.txt 中提取总体准确率（百分比）。
    优先匹配:
      Weighted average accuracy: 0.3233 (32.33%)
    """
    if not os.path.exists(score_path):
        return None

    with open(score_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    m = re.search(
        r"Weighted average accuracy:\s*[0-9.]+\s*\(([0-9.]+)%\)",
        text,
        flags=re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-list", required=True, help="run_list_minimal.yaml 路径")
    parser.add_argument("--results-dir", required=True, help="smoke 结果目录")
    parser.add_argument("--out-json", required=True, help="汇总 JSON 输出路径")
    parser.add_argument("--out-md", required=True, help="汇总 Markdown 输出路径")
    args = parser.parse_args()

    with open(args.run_list, "r", encoding="utf-8") as f:
        runs = yaml.safe_load(f).get("runs", [])

    rows = []
    for item in runs:
        out_name = item["output_name"]
        out_path = os.path.join(args.results_dir, out_name)
        m = summarize_one(out_path)
        base_no_ext = os.path.splitext(out_name)[0]
        score_path = os.path.join(args.results_dir, f"{base_no_ext}_score.txt")
        accuracy = parse_score_accuracy(score_path)
        row = {
            "series": item["series"],
            "size": item["size"],
            "quant": item["quant"],
            "env": item["env"],
            "run_enabled": item["run_enabled"],
            "skip_reason": item.get("skip_reason", ""),
            "output": out_name,
            "accuracy": accuracy,
            **m,
        }
        if not item.get("run_enabled", False):
            row["pass"] = False
            row["reason"] = row["skip_reason"] or "run_disabled"
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Smoke20 计时与质量汇总\n\n")
        f.write("| 模型系列 | 参数 | 量化 | 环境 | 已执行 | 条数 | Accuracy | 有效率 | 错误数 | 平均时延(s) | P50(s) | P95(s) | 吞吐(item/s) | 通过 | 备注 |\n")
        f.write("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            note = r.get("reason", "")
            acc_str = "-" if r.get("accuracy") is None else f"{r['accuracy']:.2f}%"
            f.write(
                f"| {r['series']} | {r['size']} | {r['quant']} | {r['env']} | "
                f"{'Y' if r['exists'] else 'N'} | {r['total']} | {acc_str} | {r['valid_rate']*100:.1f}% | "
                f"{r['error_count']} | {r['avg_latency']:.3f} | {r['p50_latency']:.3f} | "
                f"{r['p95_latency']:.3f} | {r['throughput']:.2f} | {'PASS' if r['pass'] else 'FAIL'} | {note} |\n"
            )

    print(f"[OK] JSON 汇总: {args.out_json}")
    print(f"[OK] Markdown 汇总: {args.out_md}")


if __name__ == "__main__":
    main()
