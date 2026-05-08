#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/e/Projects/eriq-eval"
STEP_DIR="$PROJECT_ROOT/stepbystep"
AUTO_CFG_DIR="$STEP_DIR/configs/quant"
RESULTS_DIR="$PROJECT_ROOT/results/smoke20_minimal"
RUN_LIST="$AUTO_CFG_DIR/run_list_minimal.yaml"
RUNNER="$STEP_DIR/engine/run_from_config.py"
EVAL_SCRIPT="$PROJECT_ROOT/eval_code/eval_hf.py"
SUMMARY_PY="$STEP_DIR/tools/summarize_smoke_metrics.py"

mkdir -p "$RESULTS_DIR"

echo "[1/4] 生成最小参数多量化 YAML..."
python "$STEP_DIR/configs/generators/generate_minimal_quant_configs.py"

if [ ! -f "$RUN_LIST" ]; then
  echo "[错误] run list 不存�? $RUN_LIST"
  exit 1
fi

echo "[2/4] 执行 smoke20（按环境隔离�?.."
source "$(conda info --base)/etc/profile.d/conda.sh"

python - <<'PY'
import os, yaml, json, subprocess, shlex

project_root = "/mnt/e/Projects/eriq-eval"
step_dir = os.path.join(project_root, "stepbystep")
results_dir = os.path.join(project_root, "results", "smoke20_minimal")
run_list = os.path.join(step_dir, "configs/quant", "run_list_minimal.yaml")
runner = os.path.join(step_dir, "engine/run_from_config.py")
eval_script = os.path.join(project_root, "eval_code", "eval_hf.py")

with open(run_list, "r", encoding="utf-8") as f:
    runs = yaml.safe_load(f).get("runs", [])

env_groups = {}
for r in runs:
    if not r.get("run_enabled", False):
        print(f"[跳过] {r['series']}-{r['size']}-{r['quant']} run_enabled=false reason={r.get('skip_reason','')}")
        continue
    env_groups.setdefault(r["env"], []).append(r)

if not env_groups:
    print("[提示] 无可执行项（run_enabled=false�?)
    raise SystemExit(0)

# 仅检查环境是否存在，不自动降级到其他环境，严格执行隔离规�?env_list_out = subprocess.check_output(["bash", "-lc", "conda info --envs"], text=True)
available_envs = set()
for line in env_list_out.splitlines():
    parts = line.strip().split()
    if not parts or parts[0].startswith("#"):
        continue
    name = parts[0]
    if name in {"*", "+"} and len(parts) > 1:
        name = parts[1]
    available_envs.add(name)

for env, items in env_groups.items():
    if env not in available_envs:
        print(f"[跳过] 环境不存�? {env}，该�?{len(items)} 个任务全部跳�?)
        continue

    for item in items:
        cfg = item["config"]
        out_name = item["output_name"]
        out_path = os.path.join(results_dir, out_name)
        score_path = out_path.replace(".json", "_score.txt")

        cmd = (
            f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
            f"conda activate {shlex.quote(env)} && "
            f"python {shlex.quote(runner)} --config {shlex.quote(cfg)} --limit 20 --output {shlex.quote(out_path)}"
        )
        print(f"[运行] env={env} config={os.path.basename(cfg)}")
        rc = subprocess.call(["bash", "-lc", cmd])
        if rc != 0:
            print(f"[失败] 推理失败: {cfg}")
            continue

        eval_cmd = f"python {shlex.quote(eval_script)} {shlex.quote(out_path)} > {shlex.quote(score_path)} 2>&1"
        rc = subprocess.call(["bash", "-lc", f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate {shlex.quote(env)} && {eval_cmd}"])
        if rc != 0:
            print(f"[警告] 评分失败: {out_name}")
        else:
            print(f"[完成] {out_name}")
PY

echo "[3/4] 汇总计时与通过状�?.."
python "$SUMMARY_PY" \
  --run-list "$RUN_LIST" \
  --results-dir "$RESULTS_DIR" \
  --out-json "$RESULTS_DIR/summary.json" \
  --out-md "$RESULTS_DIR/summary.md"

echo "[4/4] 完成"
echo "结果目录: $RESULTS_DIR"
echo "汇总报�? $RESULTS_DIR/summary.md"
