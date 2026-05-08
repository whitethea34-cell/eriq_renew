#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/e/Projects/eriq-eval"
STEP_DIR="$PROJECT_ROOT/stepbystep"

# 后台标准日志模式：不覆盖原日志，输出�?logs/standard/
if [[ "${1:-}" == "--background" ]]; then
  shift
  python "$STEP_DIR/background_log_runner.py" \
    --name run_smoke20_minimal_int8 \
    --log-dir "$PROJECT_ROOT/logs/standard" \
    --max-mb 100 \
    --backup-count 14 \
    -- bash "$0" "$@"
  exit $?
fi

ALLOW_CPU=0
if [[ "${1:-}" == "--allow-cpu" ]]; then
  ALLOW_CPU=1
fi

AUTO_CFG_DIR="$STEP_DIR/configs/quant"
RESULTS_DIR="$PROJECT_ROOT/results/smoke20_minimal_int8"
RUN_LIST="$AUTO_CFG_DIR/run_list_minimal_int8.yaml"
RUNNER="$STEP_DIR/engine/run_from_config.py"
EVAL_SCRIPT="$PROJECT_ROOT/eval_code/eval_hf.py"
SUMMARY_PY="$STEP_DIR/tools/summarize_smoke_metrics.py"
SETUP_ENVS_SH="$STEP_DIR/scripts/setup_envs.sh"

mkdir -p "$RESULTS_DIR"

echo "[1/4] 生成最小参�?INT8 YAML..."
python "$STEP_DIR/configs/generators/generate_minimal_int8_configs.py"

echo "[1.5/4] 准备系列隔离环境..."
bash "$SETUP_ENVS_SH"

if [ ! -f "$RUN_LIST" ]; then
  echo "[错误] run list 不存�? $RUN_LIST"
  exit 1
fi

echo "[预检] 检�?CUDA 可用�?.."
if ! nvidia-smi >/tmp/eriq_nvidia_smi.log 2>&1; then
  if [[ "$ALLOW_CPU" -eq 0 ]]; then
    echo "[错误] 当前会话无法访问 GPU（nvidia-smi 失败）�?
    echo "       常见原因: OS/容器策略阻止 GPU 透传�?
    echo "       详情: /tmp/eriq_nvidia_smi.log"
    echo "       如确需 CPU 慢速跑，请加参�? --allow-cpu"
    exit 2
  else
    echo "[警告] nvidia-smi 不可用，�?--allow-cpu 继续�?
  fi
fi

if [[ "$ALLOW_CPU" -eq 0 ]]; then
  if ! python - <<'PY'
import torch, sys
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
sys.exit(0 if ok else 1)
PY
  then
    echo "[错误] torch.cuda.is_available()=False，停止运行以避免 CPU 超慢冒烟�?
    echo "       如确需 CPU 慢速跑，请加参�? --allow-cpu"
    exit 2
  fi
fi

echo "[2/4] 执行 INT8 smoke20（按环境隔离�?.."
source "$(conda info --base)/etc/profile.d/conda.sh"

python - <<'PY'
import os, yaml, subprocess, shlex

project_root = "/mnt/e/Projects/eriq-eval"
step_dir = os.path.join(project_root, "stepbystep")
results_dir = os.path.join(project_root, "results", "smoke20_minimal_int8")
run_list = os.path.join(step_dir, "configs/quant", "run_list_minimal_int8.yaml")
runner = os.path.join(step_dir, "engine/run_from_config.py")
eval_script = os.path.join(project_root, "eval_code", "eval_hf.py")

with open(run_list, "r", encoding="utf-8") as f:
    runs = yaml.safe_load(f).get("runs", [])

env_groups = {}
for r in runs:
    if not r.get("run_enabled", False):
        print(f"[跳过] {r['series']}-{r['size']}-int8 run_enabled=false reason={r.get('skip_reason','')}")
        continue
    env_groups.setdefault(r["env"], []).append(r)

env_list_out = subprocess.check_output(["bash", "-lc", "conda info --envs"], text=True)
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

echo "[3/4] 汇�?INT8 计时与通过状�?.."
python "$SUMMARY_PY" \
  --run-list "$RUN_LIST" \
  --results-dir "$RESULTS_DIR" \
  --out-json "$RESULTS_DIR/summary.json" \
  --out-md "$RESULTS_DIR/summary.md"

echo "[4/4] 完成"
echo "结果目录: $RESULTS_DIR"
echo "汇总报�? $RESULTS_DIR/summary.md"
