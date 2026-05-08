#!/usr/bin/env python3
"""
ERIQ 智能失败重跑守护进程

能力:
- 自动识别失败任务（summary.json + 历史日志 + 输出缺失�?- 分批处理与并发控�?- 断点续跑（状态持久化�?- 详细重跑日志（开�?结束/失败原因/耗时�?- 后台守护进程运行（start/stop/status�?- 日志轮转与清�?- 实时监控接口（HTTP JSON�?- 系统重启后自动恢复（通过 install-service + 状态恢复）
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import pathlib
import queue
import re
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


PROJECT_ROOT_DEFAULT = "/mnt/e/Projects/eriq-eval"
CONFIG_DEFAULT = "/mnt/e/Projects/eriq-eval/stepbystep/smart_retry_daemon.yaml"


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return copy.deepcopy(default)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class GracefulStop:
    def __init__(self) -> None:
        self._stop = threading.Event()
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:
        del signum, frame
        self._stop.set()

    def is_set(self) -> bool:
        return self._stop.is_set()


class StateStore:
    def __init__(self, state_file: str, lock: threading.Lock):
        self.state_file = state_file
        self.lock = lock
        self.default_state = {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "phase": "idle",
            "tasks": {},
            "pending_queue": [],
            "in_progress": [],
            "completed": [],
            "failed": [],
            "meta": {
                "last_run_started_at": None,
                "last_run_finished_at": None,
                "last_error": None,
            },
        }

    def load(self) -> Dict[str, Any]:
        state = read_json(self.state_file, self.default_state)
        state.setdefault("tasks", {})
        state.setdefault("pending_queue", [])
        state.setdefault("in_progress", [])
        state.setdefault("completed", [])
        state.setdefault("failed", [])
        state.setdefault("meta", {})
        state["updated_at"] = now_iso()
        return state

    def save(self, state: Dict[str, Any]) -> None:
        with self.lock:
            state["updated_at"] = now_iso()
            write_json(self.state_file, state)


class EventWriter:
    def __init__(self, event_file: str, lock: threading.Lock):
        self.event_file = event_file
        self.lock = lock
        ensure_dir(os.path.dirname(event_file))

    def append(self, event: Dict[str, Any]) -> None:
        payload = {"ts": now_iso(), **event}
        line = json.dumps(payload, ensure_ascii=False)
        with self.lock:
            with open(self.event_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail(self, n: int = 200) -> List[Dict[str, Any]]:
        if not os.path.exists(self.event_file):
            return []
        with open(self.event_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n:]
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items


class MonitorServer:
    def __init__(
        self,
        host: str,
        port: int,
        state_store: StateStore,
        event_writer: EventWriter,
        logger: logging.Logger,
    ):
        self.host = host
        self.port = port
        self.state_store = state_store
        self.event_writer = event_writer
        self.logger = logger
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[ThreadingHTTPServer] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        state_store = self.state_store
        event_writer = self.event_writer

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, code: int, data: Any) -> None:
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == "/health":
                    self._send_json(200, {"ok": True, "time": now_iso()})
                    return
                if self.path == "/state":
                    self._send_json(200, state_store.load())
                    return
                if self.path.startswith("/events"):
                    m = re.search(r"limit=(\d+)", self.path)
                    limit = int(m.group(1)) if m else 100
                    self._send_json(200, event_writer.tail(limit))
                    return
                self._send_json(404, {"ok": False, "error": "not_found"})

            def log_message(self, format, *args):
                del format, args
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.logger.info("[monitor] HTTP monitor started at http://%s:%s", self.host, self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def _gzip_rotator(source: str, dest: str) -> None:
    gz_dest = f"{dest}.gz"
    with open(source, "rb") as src_f, gzip.open(gz_dest, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f)
    os.remove(source)


def _build_rotating_handler(log_file: str, max_mb: int, backup_count: int, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.namer = lambda name: name
    handler.rotator = _gzip_rotator
    return handler


def setup_logger(log_file: str, error_log_file: str, max_mb: int, backup_count: int) -> logging.Logger:
    ensure_dir(os.path.dirname(log_file))
    ensure_dir(os.path.dirname(error_log_file))
    logger = logging.getLogger("smart_retry_daemon")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | pid=%(process)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 主日�? INFO/WARNING
    handler = _build_rotating_handler(log_file, max_mb=max_mb, backup_count=backup_count, level=logging.INFO)
    handler.addFilter(MaxLevelFilter(logging.WARNING))
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    # 错误日志: ERROR+
    err_handler = _build_rotating_handler(
        error_log_file,
        max_mb=max_mb,
        backup_count=backup_count,
        level=logging.ERROR,
    )
    err_handler.setFormatter(fmt)
    logger.addHandler(err_handler)
    return logger


def cleanup_old_logs(log_dir: str, keep_days: int, logger: logging.Logger) -> None:
    if keep_days <= 0:
        return
    if not os.path.isdir(log_dir):
        return
    now = time.time()
    ttl = keep_days * 86400
    removed = 0
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            if now - os.path.getmtime(path) > ttl:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("[log] cleaned old logs: %s files", removed)


def task_id_of(item: Dict[str, Any]) -> str:
    return f"{item.get('series')}|{item.get('size')}|{item.get('quant', 'fp16')}"


def build_task_maps(run_list_file: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    data = read_yaml(run_list_file)
    runs = data.get("runs", [])
    id_to_item: Dict[str, Dict[str, Any]] = {}
    cfg_to_id: Dict[str, str] = {}
    for item in runs:
        tid = task_id_of(item)
        id_to_item[tid] = item
        cfg_to_id[os.path.abspath(item.get("config", ""))] = tid
    return id_to_item, cfg_to_id


def detect_failed_from_summary(summary_json: str) -> Set[str]:
    failed: Set[str] = set()
    rows = read_json(summary_json, [])
    for row in rows:
        passed = bool(row.get("pass"))
        reason = str(row.get("reason", "")).strip()
        exists = bool(row.get("exists"))
        if (not passed) or (reason in {"missing_file", "failed_rule"}) or (not exists):
            failed.add(f"{row.get('series')}|{row.get('size')}|{row.get('quant', 'fp16')}")
    return failed


def detect_failed_from_log(log_file: str, cfg_to_id: Dict[str, str]) -> Set[str]:
    failed: Set[str] = set()
    if not os.path.exists(log_file):
        return failed
    fail_cfg_re = re.compile(r"\[FAIL\]\s+inference:\s+(.+)")
    run_re = re.compile(r"\[RUN\]\s+([^\s]+)")
    sample_error_re = re.compile(r"\[错误\]")
    current_id: Optional[str] = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = run_re.search(line)
            if m:
                token = m.group(1)
                if "-" in token:
                    parts = token.split("-")
                    if len(parts) >= 2:
                        series = "-".join(parts[:-1])
                        size = parts[-1]
                        current_id = f"{series}|{size}|fp16"
                continue

            m = fail_cfg_re.search(line)
            if m:
                cfg_path = os.path.abspath(m.group(1).strip())
                tid = cfg_to_id.get(cfg_path)
                if tid:
                    failed.add(tid)
                continue

            if sample_error_re.search(line) and current_id:
                failed.add(current_id)
    return failed


def detect_missing_outputs(id_to_item: Dict[str, Dict[str, Any]], results_dir: str) -> Set[str]:
    missing: Set[str] = set()
    for tid, item in id_to_item.items():
        out_name = item.get("output_name", "")
        out_path = os.path.join(results_dir, out_name)
        if not os.path.exists(out_path):
            missing.add(tid)
    return missing


def run_one_task(
    task: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    project_root = config["project_root"]
    runner = config["runner_script"]
    eval_script = config["eval_script"]
    data_dir = config["data_dir"]
    limit = int(config["limit"])
    env_name = task["env"]
    cfg = task["config"]
    output_path = os.path.join(config["results_dir"], task["output_name"])
    score_path = output_path.replace(".json", "_score.txt")

    env_exports = config.get("env_exports", {})
    export_cmd = " ".join(
        f"export {k}={shlex.quote(str(v))} &&" for k, v in env_exports.items()
    )

    start = time.time()
    infer_cmd = (
        f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {shlex.quote(env_name)} && "
        f"{export_cmd} "
        f"python {shlex.quote(runner)} --config {shlex.quote(cfg)} "
        f"--limit {limit} --data-dir {shlex.quote(data_dir)} "
        f"--output {shlex.quote(output_path)}"
    )
    logger.info("[task] run start: %s (%s)", task_id_of(task), os.path.basename(cfg))
    infer_rc = subprocess.call(["bash", "-lc", infer_cmd])
    if infer_rc != 0:
        return {
            "ok": False,
            "phase": "inference",
            "return_code": infer_rc,
            "reason": f"inference_exit_{infer_rc}",
            "elapsed_sec": round(time.time() - start, 3),
            "output_path": output_path,
            "score_path": score_path,
        }

    eval_cmd = (
        f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {shlex.quote(env_name)} && "
        f"python {shlex.quote(eval_script)} {shlex.quote(output_path)} > {shlex.quote(score_path)} 2>&1"
    )
    eval_rc = subprocess.call(["bash", "-lc", eval_cmd])
    if eval_rc != 0:
        return {
            "ok": False,
            "phase": "eval",
            "return_code": eval_rc,
            "reason": f"eval_exit_{eval_rc}",
            "elapsed_sec": round(time.time() - start, 3),
            "output_path": output_path,
            "score_path": score_path,
        }

    return {
        "ok": True,
        "phase": "done",
        "return_code": 0,
        "reason": "ok",
        "elapsed_sec": round(time.time() - start, 3),
        "output_path": output_path,
        "score_path": score_path,
    }


def chunked(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


class RetryDaemon:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.stopper = GracefulStop()
        self.lock = threading.Lock()
        self.logger = setup_logger(
            cfg["daemon_log_file"],
            cfg["error_log_file"],
            int(cfg["log_rotation"]["max_mb"]),
            int(cfg["log_rotation"]["backup_count"]),
        )
        cleanup_old_logs(
            cfg["log_dir"],
            int(cfg["log_rotation"].get("cleanup_days", 14)),
            self.logger,
        )
        self.state_store = StateStore(cfg["state_file"], self.lock)
        self.event_writer = EventWriter(cfg["event_file"], self.lock)
        self.monitor = MonitorServer(
            cfg["monitor"]["host"],
            int(cfg["monitor"]["port"]),
            self.state_store,
            self.event_writer,
            self.logger,
        )

    def build_retry_queue(self) -> List[str]:
        id_to_item, cfg_to_id = build_task_maps(self.cfg["run_list_file"])
        failed: Set[str] = set()

        if self.cfg["detection"].get("from_summary", True):
            failed |= detect_failed_from_summary(self.cfg["summary_json_file"])
        if self.cfg["detection"].get("from_history_log", True):
            failed |= detect_failed_from_log(self.cfg["history_log_file"], cfg_to_id)
        if self.cfg["detection"].get("from_missing_output", True):
            failed |= detect_missing_outputs(id_to_item, self.cfg["results_dir"])

        include = set(self.cfg.get("include_tasks", []))
        exclude = set(self.cfg.get("exclude_tasks", []))
        if include:
            failed &= include
        if exclude:
            failed -= exclude

        # 仅保�?run_enabled=true 的任�?        queue_ids = []
        for tid in sorted(failed):
            item = id_to_item.get(tid)
            if not item:
                continue
            if not item.get("run_enabled", False):
                continue
            queue_ids.append(tid)
        return queue_ids

    def _bootstrap_state(self) -> Dict[str, Any]:
        state = self.state_store.load()
        id_to_item, _ = build_task_maps(self.cfg["run_list_file"])
        for tid, item in id_to_item.items():
            task = state["tasks"].setdefault(
                tid,
                {
                    "series": item.get("series"),
                    "size": str(item.get("size")),
                    "quant": item.get("quant", "fp16"),
                    "config": item.get("config"),
                    "env": item.get("env"),
                    "output_name": item.get("output_name"),
                    "status": "unknown",
                    "attempts": 0,
                    "last_start": None,
                    "last_end": None,
                    "last_reason": None,
                    "last_elapsed_sec": None,
                },
            )
            task["series"] = item.get("series")
            task["size"] = str(item.get("size"))
            task["quant"] = item.get("quant", "fp16")
            task["config"] = item.get("config")
            task["env"] = item.get("env")
            task["output_name"] = item.get("output_name")

        # 崩溃恢复：in_progress 任务回退 pending
        for tid in list(state.get("in_progress", [])):
            task = state["tasks"].get(tid)
            if task:
                task["status"] = "pending"
                task["last_reason"] = "recovered_from_unfinished_after_restart"
            if tid not in state["pending_queue"]:
                state["pending_queue"].append(tid)
        state["in_progress"] = []

        new_queue = self.build_retry_queue()
        for tid in new_queue:
            if tid not in state["pending_queue"] and tid not in state["completed"]:
                state["pending_queue"].append(tid)
                state["tasks"][tid]["status"] = "pending"

        state["phase"] = "running"
        state["meta"]["last_run_started_at"] = now_iso()
        self.state_store.save(state)
        return state

    def _update_task_state(
        self,
        state: Dict[str, Any],
        tid: str,
        status: str,
        reason: Optional[str] = None,
        elapsed_sec: Optional[float] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        task = state["tasks"][tid]
        task["status"] = status
        if reason is not None:
            task["last_reason"] = reason
        if elapsed_sec is not None:
            task["last_elapsed_sec"] = elapsed_sec
        if start is not None:
            task["last_start"] = start
        if end is not None:
            task["last_end"] = end
        self.state_store.save(state)

    def run(self) -> int:
        if self.cfg["monitor"].get("enabled", True):
            self.monitor.start()

        state = self._bootstrap_state()
        id_to_item, _ = build_task_maps(self.cfg["run_list_file"])
        max_attempts = int(self.cfg["retry"].get("max_attempts", 2))
        batch_size = int(self.cfg["batch"].get("size", 4))
        max_concurrent = int(self.cfg["batch"].get("max_concurrent", 1))
        sleep_between_batches = int(self.cfg["batch"].get("sleep_seconds", 3))

        self.logger.info(
            "[daemon] started. pending=%s, batch_size=%s, max_concurrent=%s",
            len(state["pending_queue"]),
            batch_size,
            max_concurrent,
        )
        self.event_writer.append(
            {
                "type": "daemon_start",
                "pending": len(state["pending_queue"]),
                "batch_size": batch_size,
                "max_concurrent": max_concurrent,
            }
        )

        while state["pending_queue"] and not self.stopper.is_set():
            current_batch_ids = state["pending_queue"][:batch_size]
            state["pending_queue"] = state["pending_queue"][batch_size:]
            self.state_store.save(state)

            # 过滤超重试上限任�?            runnable_ids = []
            for tid in current_batch_ids:
                attempts = int(state["tasks"][tid].get("attempts", 0))
                if attempts >= max_attempts:
                    state["tasks"][tid]["status"] = "failed_permanent"
                    state["tasks"][tid]["last_reason"] = f"max_attempts_exceeded({max_attempts})"
                    if tid not in state["failed"]:
                        state["failed"].append(tid)
                    self.event_writer.append(
                        {"type": "task_skip", "task_id": tid, "reason": "max_attempts_exceeded"}
                    )
                else:
                    runnable_ids.append(tid)
            self.state_store.save(state)

            if not runnable_ids:
                continue

            with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
                futures = {}
                for tid in runnable_ids:
                    task = id_to_item[tid]
                    start_iso = now_iso()
                    state["tasks"][tid]["attempts"] = int(state["tasks"][tid].get("attempts", 0)) + 1
                    self._update_task_state(state, tid, "in_progress", start=start_iso)
                    if tid not in state["in_progress"]:
                        state["in_progress"].append(tid)
                    self.state_store.save(state)
                    self.event_writer.append(
                        {
                            "type": "task_start",
                            "task_id": tid,
                            "attempt": state["tasks"][tid]["attempts"],
                            "config": task["config"],
                            "env": task["env"],
                        }
                    )
                    futures[ex.submit(run_one_task, task, self.cfg, self.logger)] = (tid, task, start_iso)

                for fut in as_completed(futures):
                    tid, task, start_iso = futures[fut]
                    end_iso = now_iso()
                    if tid in state["in_progress"]:
                        state["in_progress"].remove(tid)
                    try:
                        result = fut.result()
                    except Exception as e:  # noqa: BLE001
                        result = {
                            "ok": False,
                            "phase": "runtime_exception",
                            "return_code": -1,
                            "reason": f"exception:{e}",
                            "elapsed_sec": None,
                        }

                    if result["ok"]:
                        self._update_task_state(
                            state,
                            tid,
                            "completed",
                            reason="ok",
                            elapsed_sec=result.get("elapsed_sec"),
                            end=end_iso,
                        )
                        if tid not in state["completed"]:
                            state["completed"].append(tid)
                        self.event_writer.append(
                            {
                                "type": "task_end",
                                "task_id": tid,
                                "status": "completed",
                                "start": start_iso,
                                "end": end_iso,
                                "elapsed_sec": result.get("elapsed_sec"),
                                "reason": "ok",
                                "output_path": result.get("output_path"),
                                "score_path": result.get("score_path"),
                            }
                        )
                        self.logger.info("[task] success: %s", tid)
                    else:
                        reason = result.get("reason", "unknown_fail")
                        self._update_task_state(
                            state,
                            tid,
                            "failed",
                            reason=reason,
                            elapsed_sec=result.get("elapsed_sec"),
                            end=end_iso,
                        )
                        if tid not in state["failed"]:
                            state["failed"].append(tid)
                        attempts = int(state["tasks"][tid].get("attempts", 0))
                        if attempts < max_attempts:
                            state["pending_queue"].append(tid)
                            state["tasks"][tid]["status"] = "pending"
                        self.event_writer.append(
                            {
                                "type": "task_end",
                                "task_id": tid,
                                "status": "failed",
                                "start": start_iso,
                                "end": end_iso,
                                "elapsed_sec": result.get("elapsed_sec"),
                                "reason": reason,
                                "phase": result.get("phase"),
                                "return_code": result.get("return_code"),
                            }
                        )
                        self.logger.warning("[task] failed: %s, reason=%s", tid, reason)
                    self.state_store.save(state)

            if self.stopper.is_set():
                break
            if sleep_between_batches > 0:
                time.sleep(sleep_between_batches)

        state["phase"] = "stopped" if self.stopper.is_set() else "completed"
        state["meta"]["last_run_finished_at"] = now_iso()
        self.state_store.save(state)
        self.event_writer.append(
            {
                "type": "daemon_end",
                "phase": state["phase"],
                "pending": len(state["pending_queue"]),
                "completed": len(state["completed"]),
                "failed": len(state["failed"]),
            }
        )
        self.logger.info(
            "[daemon] end phase=%s, pending=%s, completed=%s, failed=%s",
            state["phase"],
            len(state["pending_queue"]),
            len(state["completed"]),
            len(state["failed"]),
        )
        self.monitor.stop()
        return 0


def build_runtime_config(config_file: str) -> Dict[str, Any]:
    cfg = read_yaml(config_file)
    project_root = cfg.get("project_root", PROJECT_ROOT_DEFAULT)
    log_dir = cfg.get("log_dir", os.path.join(project_root, "logs", "standard", "smart_retry"))
    state_dir = cfg.get("state_dir", os.path.join(project_root, "runtime", "smart_retry"))
    ensure_dir(log_dir)
    ensure_dir(state_dir)

    runtime = {
        "project_root": project_root,
        "run_list_file": cfg.get(
            "run_list_file",
            os.path.join(project_root, "stepbystep", "configs", "run_lists", "run_list_fp16_budget.yaml"),
        ),
        "summary_json_file": cfg.get(
            "summary_json_file",
            os.path.join(project_root, "results", "smoke20_minimal_fp16_20b", "summary.json"),
        ),
        "history_log_file": cfg.get(
            "history_log_file",
            os.path.join(project_root, "logs", "retry_failed_models_fp16_20b.log"),
        ),
        "results_dir": cfg.get(
            "results_dir",
            os.path.join(project_root, "results", "smoke20_minimal_fp16_20b"),
        ),
        "runner_script": cfg.get(
            "runner_script",
            os.path.join(project_root, "stepbystep", "run_from_config_per_subset.py"),
        ),
        "eval_script": cfg.get(
            "eval_script",
            os.path.join(project_root, "eval_code", "eval_hf.py"),
        ),
        "data_dir": cfg.get("data_dir", os.path.join(project_root, "data")),
        "limit": int(cfg.get("limit", 20)),
        "batch": cfg.get("batch", {"size": 4, "max_concurrent": 1, "sleep_seconds": 3}),
        "retry": cfg.get("retry", {"max_attempts": 2}),
        "detection": cfg.get(
            "detection",
            {"from_summary": True, "from_history_log": True, "from_missing_output": True},
        ),
        "include_tasks": cfg.get("include_tasks", []),
        "exclude_tasks": cfg.get("exclude_tasks", []),
        "env_exports": cfg.get(
            "env_exports",
            {
                "ERIQ_SAFE_ATTN": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "ERIQ_SAMPLE_EMPTY_CACHE": "1",
                "ERIQ_GC_EACH_SAMPLE": "1",
                "ERIQ_SUBSET_COOLDOWN_SEC": "3",
                "CUDA_LAUNCH_BLOCKING": "0",
            },
        ),
        "log_dir": log_dir,
        "state_dir": state_dir,
        "daemon_log_file": cfg.get("daemon_log_file", os.path.join(log_dir, "smart_retry_daemon.log")),
        "error_log_file": cfg.get("error_log_file", os.path.join(log_dir, "smart_retry_daemon.error.log")),
        "event_file": cfg.get("event_file", os.path.join(log_dir, "rerun_events.jsonl")),
        "state_file": cfg.get("state_file", os.path.join(state_dir, "retry_state.json")),
        "pid_file": cfg.get("pid_file", os.path.join(state_dir, "retry_daemon.pid")),
        "log_rotation": cfg.get("log_rotation", {"max_mb": 50, "backup_count": 10, "cleanup_days": 14}),
        "monitor": cfg.get("monitor", {"enabled": True, "host": "127.0.0.1", "port": 8765}),
        "service_name": cfg.get("service_name", "eriq-smart-retry"),
    }
    return runtime


def read_pid(pid_file: str) -> Optional[int]:
    if not os.path.exists(pid_file):
        return None
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_pid(pid_file: str, pid: int) -> None:
    ensure_dir(os.path.dirname(pid_file))
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(pid))


def remove_pid(pid_file: str) -> None:
    try:
        os.remove(pid_file)
    except OSError:
        pass


def cmd_run(cfg: Dict[str, Any]) -> int:
    pid_file = cfg["pid_file"]
    write_pid(pid_file, os.getpid())
    try:
        daemon = RetryDaemon(cfg)
        return daemon.run()
    finally:
        remove_pid(pid_file)


def cmd_start(cfg: Dict[str, Any], config_file: str) -> int:
    pid = read_pid(cfg["pid_file"])
    if pid and is_process_alive(pid):
        print(f"[start] daemon already running, pid={pid}")
        return 0
    log_file = cfg["daemon_log_file"]
    ensure_dir(os.path.dirname(log_file))
    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, os.path.abspath(__file__), "run", "--config", config_file],
            stdout=lf,
            stderr=lf,
            start_new_session=True,
            close_fds=True,
        )
    write_pid(cfg["pid_file"], proc.pid)
    print(f"[start] daemon started, pid={proc.pid}, log={log_file}")
    return 0


def cmd_stop(cfg: Dict[str, Any], timeout_sec: int = 20) -> int:
    pid = read_pid(cfg["pid_file"])
    if not pid or not is_process_alive(pid):
        remove_pid(cfg["pid_file"])
        print("[stop] daemon is not running")
        return 0
    print(f"[stop] stopping daemon pid={pid}")
    os.kill(pid, signal.SIGTERM)
    end = time.time() + timeout_sec
    while time.time() < end:
        if not is_process_alive(pid):
            remove_pid(cfg["pid_file"])
            print("[stop] daemon stopped")
            return 0
        time.sleep(0.5)
    print("[stop] timeout, send SIGKILL")
    os.kill(pid, signal.SIGKILL)
    remove_pid(cfg["pid_file"])
    return 0


def cmd_status(cfg: Dict[str, Any]) -> int:
    pid = read_pid(cfg["pid_file"])
    alive = bool(pid and is_process_alive(pid))
    state = read_json(cfg["state_file"], {})
    print(
        json.dumps(
            {
                "running": alive,
                "pid": pid if alive else None,
                "phase": state.get("phase"),
                "pending": len(state.get("pending_queue", [])),
                "in_progress": len(state.get("in_progress", [])),
                "completed": len(state.get("completed", [])),
                "failed": len(state.get("failed", [])),
                "state_file": cfg["state_file"],
                "log_file": cfg["daemon_log_file"],
                "monitor": cfg["monitor"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_tail(cfg: Dict[str, Any], follow: bool, lines: int) -> int:
    log_file = cfg["daemon_log_file"]
    if not os.path.exists(log_file):
        print(f"[tail] log not found: {log_file}")
        return 1
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        data = f.readlines()
        for line in data[-lines:]:
            print(line.rstrip("\n"))
        if not follow:
            return 0
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.5)
                f.seek(where)
            else:
                print(line.rstrip("\n"))


def cmd_install_service(cfg: Dict[str, Any], config_file: str) -> int:
    service_name = cfg["service_name"]
    service_path = f"/etc/systemd/system/{service_name}.service"
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    unit = f"""[Unit]
Description=ERIQ Smart Retry Daemon
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory={cfg["project_root"]}
ExecStart={python_path} {script_path} run --config {os.path.abspath(config_file)}
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
"""
    print("[service] writing service file:", service_path)
    try:
        with open(service_path, "w", encoding="utf-8") as f:
            f.write(unit)
    except PermissionError:
        print("[service] permission denied. please run with sudo.")
        return 1
    print(f"[service] created. then run:\n  sudo systemctl daemon-reload\n  sudo systemctl enable {service_name}\n  sudo systemctl start {service_name}")
    return 0


def _extract_config_arg(argv: List[str]) -> Tuple[str, List[str]]:
    config = CONFIG_DEFAULT
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 < len(argv):
                config = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--config="):
            config = arg.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return config, cleaned


def parse_args() -> Tuple[str, argparse.Namespace]:
    config, cleaned_argv = _extract_config_arg(sys.argv)
    parser = argparse.ArgumentParser(description="ERIQ 智能重跑守护进程")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="前台运行守护进程（调�?服务模式�?)
    sub.add_parser("start", help="后台启动守护进程")
    sub.add_parser("stop", help="停止守护进程")
    sub.add_parser("status", help="查看守护进程状�?)

    p_tail = sub.add_parser("tail", help="查看守护日志")
    p_tail.add_argument("-f", "--follow", action="store_true", help="持续跟踪日志")
    p_tail.add_argument("-n", "--lines", type=int, default=100, help="显示最�?N �?)

    sub.add_parser("install-service", help="安装 systemd 服务，实现重启自动恢�?)
    return config, parser.parse_args(cleaned_argv[1:])


def main() -> int:
    config_from_cli, args = parse_args()
    config_file = os.path.abspath(config_from_cli)
    cfg = build_runtime_config(config_file)

    if args.cmd == "run":
        return cmd_run(cfg)
    if args.cmd == "start":
        return cmd_start(cfg, config_file)
    if args.cmd == "stop":
        return cmd_stop(cfg)
    if args.cmd == "status":
        return cmd_status(cfg)
    if args.cmd == "tail":
        return cmd_tail(cfg, args.follow, args.lines)
    if args.cmd == "install-service":
        return cmd_install_service(cfg, config_file)
    print("unknown cmd")
    return 1


if __name__ == "__main__":
    sys.exit(main())
