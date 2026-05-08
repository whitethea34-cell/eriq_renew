#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/e/Projects/eriq-eval"
CFG="${ROOT}/stepbystep/partial-retry-full-int8/retry-partial-full-int8.yaml"
RUN_LIST="${ROOT}/stepbystep/partial-retry-full-int8/run_list_int8_current.yaml"
RES_DIR="${ROOT}/results/exp-int8-full"
STATE_DIR="${ROOT}/runtime/partial-retry-full-int8"
LOG_DIR="${ROOT}/logs/standard/partial-retry-full-int8"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${RES_DIR}"

# Stop previous daemon instances to avoid pid/port conflicts.
pkill -f "smart_retry_daemon.py run --config ${CFG}" >/dev/null 2>&1 || true
pkill -f "engine/run_from_config.py" >/dev/null 2>&1 || true

# Clean current run state and events for a fresh rerun.
rm -f "${STATE_DIR}/retry_daemon.pid" "${STATE_DIR}/retry_state.json" "${LOG_DIR}/rerun_events.jsonl"

# Clean outputs of enabled tasks only.
python - <<'PY'
import os, yaml
run_list = "/mnt/e/Projects/eriq-eval/stepbystep/partial-retry-full-int8/run_list_int8_current.yaml"
res_dir = "/mnt/e/Projects/eriq-eval/results/exp-int8-full"
with open(run_list, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
for item in data.get("runs", []):
    if not item.get("run_enabled", False):
        continue
    out = item.get("output_name")
    if not out:
        continue
    for p in [
        os.path.join(res_dir, out),
        os.path.join(res_dir, out + ".partial.json"),
        os.path.join(res_dir, out.rsplit(".json", 1)[0] + "_score.txt"),
    ]:
        if os.path.exists(p):
            os.remove(p)
            print(f"removed: {p}")
PY

python "${ROOT}/stepbystep/smart_retry_daemon.py" start --config "${CFG}"
python "${ROOT}/stepbystep/smart_retry_daemon.py" status --config "${CFG}"
python "${ROOT}/stepbystep/smart_retry_daemon.py" tail -f -n 200 --config "${CFG}"
