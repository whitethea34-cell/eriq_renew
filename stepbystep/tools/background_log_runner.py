#!/usr/bin/env python3
"""
后台命令日志代理：
- 为任意命令生成标准格式日志（时间戳/级别/进程ID/模块/消息）
- 主日志与错误日志分离
- 支持大小轮转 + 压缩归档
"""

from __future__ import annotations

import argparse
import gzip
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PROJECT_ROOT = "/mnt/e/Projects/eriq-eval"
DEFAULT_LOG_DIR = f"{DEFAULT_PROJECT_ROOT}/logs/standard"


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def _gzip_rotator(source: str, dest: str) -> None:
    gz_dest = f"{dest}.gz"
    with open(source, "rb") as sf, gzip.open(gz_dest, "wb") as df:
        shutil.copyfileobj(sf, df)
    os.remove(source)


def _build_rotating_handler(path: Path, max_mb: int, backup_count: int, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        filename=str(path),
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.namer = lambda name: name
    handler.rotator = _gzip_rotator
    return handler


def build_logger(name: str, log_dir: str, max_mb: int, backup_count: int) -> tuple[logging.Logger, Path, Path]:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"eriq.bg.{name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    info_path = Path(log_dir) / f"{name}_{ts}.log"
    err_path = Path(log_dir) / f"{name}_{ts}.error.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | pid=%(process)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    info_handler = _build_rotating_handler(info_path, max_mb=max_mb, backup_count=backup_count, level=logging.INFO)
    info_handler.addFilter(MaxLevelFilter(logging.WARNING))
    info_handler.setFormatter(fmt)
    logger.addHandler(info_handler)

    err_handler = _build_rotating_handler(err_path, max_mb=max_mb, backup_count=backup_count, level=logging.ERROR)
    err_handler.setFormatter(fmt)
    logger.addHandler(err_handler)
    return logger, info_path, err_path


_ERR_PAT = re.compile(r"(traceback|exception|error|oom|out of memory|\[fail\]|失败)", re.IGNORECASE)
_WARN_PAT = re.compile(r"(warn|warning|警告)", re.IGNORECASE)


def infer_level(line: str, stream: str) -> int:
    if _ERR_PAT.search(line):
        return logging.ERROR
    if _WARN_PAT.search(line):
        return logging.WARNING
    if stream == "stderr":
        return logging.ERROR
    return logging.INFO


def _drain_stream(
    logger: logging.Logger,
    proc: subprocess.Popen[str],
    stream_name: str,
    lines: Iterable[str],
) -> None:
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        level = infer_level(line, stream_name)
        logger.log(level, "[%s] %s", stream_name, line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后台运行命令并写入标准日志")
    parser.add_argument("--name", required=True, help="日志任务名（用于日志文件名前缀）")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="日志目录")
    parser.add_argument("--max-mb", type=int, default=100, help="单文件轮转大小（MB）")
    parser.add_argument("--backup-count", type=int, default=14, help="保留轮转文件数")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="要执行的命令，形如 -- bash xxx.sh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("缺少命令参数。示例: background_log_runner.py --name smoke -- bash run.sh")
        return 2

    logger, info_path, err_path = build_logger(args.name, args.log_dir, args.max_mb, args.backup_count)
    logger.info("后台任务启动 command=%s", " ".join(command))

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    logger.info("子进程已启动 pid=%s", proc.pid)

    out_lines = proc.stdout if proc.stdout is not None else []
    err_lines = proc.stderr if proc.stderr is not None else []
    t1 = threading.Thread(target=_drain_stream, args=(logger, proc, "stdout", out_lines), daemon=True)
    t2 = threading.Thread(target=_drain_stream, args=(logger, proc, "stderr", err_lines), daemon=True)
    t1.start()
    t2.start()

    rc = proc.wait()
    t1.join()
    t2.join()

    if rc == 0:
        logger.info("后台任务结束 exit_code=%s", rc)
    else:
        logger.error("后台任务失败 exit_code=%s", rc)

    print(f"[OK] 标准日志: {info_path}")
    print(f"[OK] 错误日志: {err_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
