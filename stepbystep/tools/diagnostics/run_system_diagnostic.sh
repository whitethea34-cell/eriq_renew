#!/usr/bin/env bash
set -euo pipefail

# ERIQ 模型系统性排查入口脚�?# 说明:
# - 该脚本用于真实环境执�?# - 不会自动启动全量重跑，只做报告生成与可选单模型断言

PROJECT_ROOT="/mnt/e/Projects/eriq-eval"
RUN_LIST="${PROJECT_ROOT}/stepbystep/configs/fp16_budget/run_list_fp16_budget.yaml"
STATE_FILE="${PROJECT_ROOT}/runtime/smart_retry/retry_state.json"
LOG1="${PROJECT_ROOT}/logs/smart_retry/smart_retry_daemon.log"
LOG2="${PROJECT_ROOT}/logs/retry_failed_models_fp16_20b.log"
OUT_MD="${PROJECT_ROOT}/stepbystep/diagnostics/system_diagnostic_report.md"
OUT_JSON="${PROJECT_ROOT}/stepbystep/diagnostics/system_diagnostic_report.json"

echo "[1/3] 生成离线系统排查报告"
python "${PROJECT_ROOT}/stepbystep/diagnostics/generate_system_diagnostic_report.py" \
  --run-list "${RUN_LIST}" \
  --state-file "${STATE_FILE}" \
  --logs "${LOG1}" "${LOG2}" \
  --out-md "${OUT_MD}" \
  --out-json "${OUT_JSON}"

echo "[2/3] (可�? 单模型静态断言示例"
python "${PROJECT_ROOT}/stepbystep/diagnostics/assert_model_pipeline.py" \
  --config "${PROJECT_ROOT}/stepbystep/configs/fp16_budget/internvl2.5-1b-fp16.yaml"

echo "[3/3] 完成"
echo "报告: ${OUT_MD}"
echo "结构化输�? ${OUT_JSON}"
