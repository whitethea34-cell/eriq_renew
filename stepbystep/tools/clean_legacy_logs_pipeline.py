#!/usr/bin/env python3
"""
ERIQ 日志清洗分档系统 (Log Pipeline Processor) - 增强版

功能目标:
1) 解析原始累积日志，支持多种时间格式（含标准文本和带管道符的格式）。
2) 降噪：自动过滤 tqdm 进度条、重复的刷新字符。
3) 分流：按日期和级别（INFO, ERROR）归档。
4) 灵活：支持命令行指定文件或目录，支持 Windows/Linux 路径。
"""

import os
import re
import datetime
import argparse
from pathlib import Path

# 正则表达式配置
# 1) 匹配多种时间格式: 2026-04-30 08:19:02,953 或 2026-05-06 15:41:08
TIME_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:,\d{3})?)")
# 2) 匹配错误关键字
ERROR_REGEX = re.compile(r"(ERROR|Exception|Traceback|CUDA out of memory|device not ready|unknown error|FAILED)", re.IGNORECASE)
# 3) 匹配进度条噪音
PROGRESS_RE = re.compile(r"(\d+%.*\||it/s|s/it|██|▏|▎|▍|▌|▋|▊|▉)")

def format_iso8601(dt_obj):
    """统一标准化为 ISO 8601 毫秒级格式"""
    return dt_obj.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+08:00"

def determine_module(path: Path):
    """基于路径和文件名判定模块名"""
    full_path = str(path).lower()
    if "smoke" in full_path:
        return "main_pipeline"
    elif "retry" in full_path:
        return "retry_daemon"
    elif "dualview" in full_path:
        return "quick_check"
    return path.stem

def process_log_file(src_path: Path, archive_root: Path):
    if not src_path.exists():
        print(f"[错误] 文件不存在: {src_path}")
        return

    module_name = determine_module(src_path)
    mtime_dt = datetime.datetime.fromtimestamp(src_path.stat().st_mtime)
    current_dt = mtime_dt
    
    print(f"[处理] 开始清洗: {src_path}")
    
    processed_count = 0
    skipped_count = 0
    
    # 临时存储清洗后的行，避免频繁 IO
    results = []

    try:
        with src_path.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                raw_line = line.strip()
                if not raw_line:
                    continue
                
                # 1) 进度条降噪
                if PROGRESS_RE.search(raw_line):
                    skipped_count += 1
                    continue

                # 2) 时间戳提取
                time_match = TIME_REGEX.search(raw_line)
                if time_match:
                    time_str = time_match.group(1)
                    try:
                        fmt = "%Y-%m-%d %H:%M:%S,%f" if "," in time_str else "%Y-%m-%d %H:%M:%S"
                        current_dt = datetime.datetime.strptime(time_str, fmt)
                        # 去掉时间戳前缀和多余的管道符
                        clean_content = raw_line[len(time_str):].strip(" |-|")
                    except ValueError:
                        clean_content = raw_line
                else:
                    clean_content = raw_line

                iso_ts = format_iso8601(current_dt)
                date_tag = current_dt.strftime("%Y-%m-%d")
                
                # 3) 级别判定
                level = "ERROR" if ERROR_REGEX.search(clean_content) else "INFO"
                
                # 4) 格式化
                structured_log = f"[{iso_ts}] [{level}] [{module_name}] {clean_content}\n"
                
                # 按天和级别存储
                target_dir = archive_root / date_tag / level
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / f"{module_name}.log"
                
                with target_file.open('a', encoding='utf-8') as out_f:
                    out_f.write(structured_log)
                
                processed_count += 1
                
    except Exception as e:
        print(f"[失败] 处理过程出错: {e}")
        return

    print(f"[完成] 清洗完毕. 保留: {processed_count} 行, 过滤噪音: {skipped_count} 行")
    print(f"       结果归档至: {archive_root}")

def main():
    parser = argparse.ArgumentParser(description="ERIQ 日志清洗与分层归档工具")
    parser.add_argument("path", help="要处理的日志文件或目录路径")
    parser.add_argument("--archive-dir", default=None, help="归档根目录 (默认为 logs/archive)")
    
    args = parser.parse_args()
    src_path = Path(args.path).resolve()
    
    # 自动定位项目 logs/archive
    if args.archive_dir:
        archive_root = Path(args.archive_dir).resolve()
    else:
        # 假设脚本在 stepbystep/tools 下，向上寻找项目根目录
        project_root = src_path.parent
        while project_root.name and project_root.name != "eriq-eval":
            project_root = project_root.parent
        if project_root.name == "eriq-eval":
            archive_root = project_root / "logs" / "archive"
        else:
            archive_root = src_path.parent / "archive"

    if src_path.is_file():
        process_log_file(src_path, archive_root)
    elif src_path.is_dir():
        for log_file in src_path.rglob("*.log"):
            if "archive" in str(log_file): continue
            process_log_file(log_file, archive_root)
    else:
        print(f"[错误] 无效路径: {src_path}")

if __name__ == "__main__":
    main()
