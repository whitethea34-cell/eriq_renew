#!/usr/bin/env bash

# ==============================================================================
# SmolVLM-512M-INT8 定时自动执行与汇总脚�?# 功能: 延迟执行 -> 激活环�?-> 子任务拆分容错推�?-> 结果清洗合并 -> 官方打分 -> 生成全局汇总报�?# 特�? 
# 1. 纯后台执行，支持通过 nohup 挂载，断网不中断
# 2. 【新增】日志分流与阅后即焚：主日志与错误日志分离，任务成功后自动删除几十万行的主日�?# 3. 容错性极强，子任务间隔离，某一条数�?OOM 不影响全局
# ==============================================================================

set -euo pipefail

ROOT="/mnt/e/Projects/eriq-eval"
LOG_DIR="$ROOT/logs/smolvlm_auto"
MAIN_LOG="$LOG_DIR/smolvlm_512m_int8_main_$(date +%Y%m%d_%H%M%S).log"
ERROR_LOG="$LOG_DIR/smolvlm_512m_int8_error_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

# 全局分流：标准输出进�?MAIN_LOG，标准错误进�?ERROR_LOG
exec > >(tee -a "$MAIN_LOG") 2> >(tee -a "$ERROR_LOG" >&2)

echo "=================================================="
echo "时间: $(date)"
echo "启动 SmolVLM-512M-INT8 自动化评测流水线"
echo "主日�? $MAIN_LOG"
echo "错误日志: $ERROR_LOG"
echo "=================================================="

# ---------------------------------------------------------
# 阶段 1: 延迟执行
# ---------------------------------------------------------
DELAY_SECONDS=$((2 * 5))
echo "[阶段 1] 等待 10 ($DELAY_SECONDS �? 以准备执行任�?.."
sleep $DELAY_SECONDS
echo "[阶段 1] 延迟结束，开始正式执行任务！ 时间: $(date)"

# ---------------------------------------------------------
# 阶段 2: 环境变量与配置准�?# ---------------------------------------------------------
echo "[阶段 2] 初始化环�?.."
CFG="$ROOT/stepbystep/configs/quant/smolvlm-512m-int8.yaml"
OUT="$ROOT/results/exp-int8-full/smolvlm-512m-int8_smoke20.json"
PART="$OUT.partial.json"
PART_DIR="$ROOT/results/exp-int8-full/resume_parts_smolvlm_512m"

mkdir -p "$PART_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eriq-smolvlm

# ---------------------------------------------------------
# 阶段 3: 子任务拆分推理（容错隔离�?# ---------------------------------------------------------
echo "[阶段 3] 开始按子任务执行模型推�?.."
SUBSETS=(
  QA_ACTION_UNDERSTANDING
  QA_DUALVIEW_MATCHING
  QA_FINE_GRAINED_PLAN
  QA_HUMAN_INTENTION
  QA_HUMAN_INTERACTION
  QA_MISTAKE_CLASSIFY
  QA_MISTAKE_EXISTENCE
  QA_MISTAKE_RECOVERY
  QA_RELATIVE_POS_GROUNDING
  QA_SCENE_UNDERSTANDING
  QA_SUBTASK_PLANNING
  QA_SUCCESS_DETECTION
  QA_TASK_GROUNDING
  QA_TASK_PROGRESS
  QA_TRAJ_UNDERSTANDING
)

for s in "${SUBSETS[@]}"; do
  echo "[RUN] $s - 时间: $(date)"
  # 使用 || true 防止某个子任务崩溃导致整个脚本退�?(set -e 的特�?
  python "$ROOT/stepbystep/engine/run_from_config.py" \
    --config "$CFG" \
    --limit 0 \
    --data-dir "$ROOT/data" \
    --subset-file "${s}.json" \
    --output "$PART_DIR/${s}.json" || echo "[严重错误] 子任�?$s 执行失败！继续执行下一�?.."
done

# ---------------------------------------------------------
# 阶段 4: 结果清洗与合�?# ---------------------------------------------------------
echo "[阶段 4] 开始清洗脏数据并合�?JSON 结果..."
python - <<PY
import json, os, glob

out = "$OUT"
part_dir = "$PART_DIR"

by_id = {}
for p in sorted(glob.glob(os.path.join(part_dir, "QA_*.json"))):
    try:
        arr = json.load(open(p, "r", encoding="utf-8"))
        for x in arr:
            # 剔除因为 OOM 或其他异常导致的无效输出
            if "ERROR" in str(x.get("raw_output", "")):
                continue
            by_id[x.get("id")] = x
    except Exception as e:
        print(f"[警告] 解析合并文件 {p} 时出�? {e}")

merged = list(by_id.values())
try:
    json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[合并完成] 输出�? {out}, 有效数据总数={len(merged)}")
except Exception as e:
    print(f"[错误] 最终写�?JSON 文件失败: {e}")
PY

# ---------------------------------------------------------
# 阶段 5: 官方打分
# ---------------------------------------------------------
echo "[阶段 5] 开始执行官方脚本打�?.."
SCORE_FILE="${OUT%.json}_score.txt"
if [ -f "$OUT" ]; then
    python "$ROOT/eval_code/eval_hf.py" "$OUT" > "$SCORE_FILE" 2>&1 || echo "[错误] 打分脚本执行失败�?
    echo "[打分完成] 得分结果已保存至: $SCORE_FILE"
    echo "========= 简要得分预�?========="
    cat "$SCORE_FILE" | tail -n 20 || true
    echo "================================"
else
    echo "[错误] 找不到合并后�?JSON 结果文件，无法打分�?
fi

# ---------------------------------------------------------
# 阶段 6: 全局汇总报告生�?# ---------------------------------------------------------
echo "[阶段 6] 触发全局报告更新..."
# 调用项目现有的阶段汇总脚本（读取 results 下的所有的文件生成 summary.md�?# README.md 不是可执行脚本，真正生成汇总的�?summarize_smoke_metrics.py
SUMMARY_SCRIPT="$ROOT/stepbystep/summarize_smoke_metrics.py"
if [ -f "$SUMMARY_SCRIPT" ]; then
    python "$SUMMARY_SCRIPT" \
        --input-dir "$ROOT/results/exp-int8-full" \
        --output-dir "$ROOT/results/exp-int8-full" || echo "[错误] 全局汇总报告生成失败！"
    echo "[汇总完成] 请检�?$ROOT/results/exp-int8-full/summary.md"
else
    echo "[警告] 未找到汇总脚�?$SUMMARY_SCRIPT，跳过报告生成�?
fi

echo "=================================================="
echo "时间: $(date)"
echo "SmolVLM-512M-INT8 自动化评测流水线全部完成�?
echo "主日�? $MAIN_LOG"
echo "错误日志: $ERROR_LOG"
echo "=================================================="

# ---------------------------------------------------------
# 阶段 7: 阅后即焚 (生命周期控制)
# ---------------------------------------------------------
# 检查最终输出的 JSON 文件是否成功生成且文件大小大�?0
if [ -s "$OUT" ]; then
    echo "[阅后即焚] 评测任务成功！正在删除庞大的主日�?.."
    rm -f "$MAIN_LOG"
    echo "[阅后即焚] $MAIN_LOG 已删除。排障请查看错误日志: $ERROR_LOG"
else
    echo "[保留] 警告：未能生成有效的输出结果 JSON，保留主日志以供排查�?
fi
