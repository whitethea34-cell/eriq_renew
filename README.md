# ERIQ 大模型评测 (ERIQ Evaluation Suite)

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-orange.svg)](https://pytorch.org/)

本仓库是一套完整的视觉语言模型 (VLM) 评测工具链，专为 **智元机器人 ERIQ (Embodied Reasoning Intelligence Quotient)** 具身推理评测集设计。支持在本地 GPU（推荐 RTX 4090/4060 等）上执行多模型、多量化版本的批量推理、自动打分与故障恢复。

***

## 🌟 设计思路

- **🚀 统一推理入口**: 通过 YAML 配置文件驱动模型切换，支持 FP16、INT8、4-bit、AWQ 等多种量化格式，无需修改 Python 代码。
- **🛡️ 环境隔离**: 为每个模型系列（如 InternVL, Qwen, MiniCPM）创建独立的 Conda 环境，完美规避 `transformers` 版本冲突。
- **♻️ 自动化故障恢复**:
  - **CUDA 异常重载**: 自动捕获 OOM 或内核异常，支持断点重续。
  - **智能重试守护 (Retry Daemon)**: 实时监控后台进程，自动补测失败模型。
- **📊 规范化日志管理**: 引入“阅后即焚”机制，自动过滤进度条噪音，分流 `INFO` 与 `ERROR`，支持毫秒级 ISO 8601 时间戳。
- **📈 一键生成报告**: 自动调用官方评分脚本，汇总 PASS/FAIL、时延、有效率、P95 等核心指标。

***

## 🛠️ 技术栈

- **语言**: [Python 3.12.10](file:///e:/Projects/eriq-eval/pyrightconfig.json)
- **深度学习**: [PyTorch 2.6.0+cu124](file:///e:/Projects/eriq-eval/stepbystep/scripts/setup_envs.sh), Transformers, Accelerate, Timm
- **量化**: Bitsandbytes (8bit/4bit), AutoAWQ
- **基础设施**: WSL2 (Ubuntu 22.04) + NVIDIA CUDA 12.4+
- **模型来源**: ModelScope (优先), HuggingFace

***

## 📂 目录结构

项目采用模块化设计，核心逻辑位于 `stepbystep/` 目录下：

```text
eriq-eval/
├── data/                       # ERIQ 数据集 (15 个 QA_*.json)
├── models/                     # 模型权重本地缓存 (ModelScope/HF)
├── results/                    # 评测输出 (JSON) 与 汇总报告 (Markdown)
├── logs/                       # 日志系统 (含归档与精简日志)
├── eval_code/                  # 官方评分引擎 (eval_hf.py)
└── stepbystep/                 # 推理与运维核心目录
    ├── engine/                 # 核心引擎 (run_from_config.py, run_from_config_per_subset.py)
    ├── configs/                # 配置体系 (Templates, Generators, Run_lists)
    │   ├── fp16_budget/        # ≤20B 显存预算配置
    │   ├── quant/              # 4bit/8bit 量化配置
    │   └── generators/         # 自动生成器脚本
    ├── scripts/                # 批量运行脚本 (Shell)
    ├── tools/                  # 运维工具 (诊断、迁移、后处理)
    └── retry/                  # 智能重试系统 (Smart Retry Daemon)
```

***

## 🚀 快速开始

### 1. 环境初始化

```bash
# 1. 克隆项目
git clone https://github.com/whitethea34-cell/eriq-eval.git
cd eriq-eval

# 2. 初始化 E 盘存储结构 (推荐)
bash stepbystep/tools/setup/setup_e_drive.sh

# 3. 批量创建隔离环境
bash stepbystep/scripts/setup_envs.sh
```

### 2. 执行评测

**方案 A: 跑全量 FP16≤20B 主流程 (推荐)**

```bash
bash stepbystep/scripts/run_smoke20_minimal_fp16_20b.sh
```

**方案 B: 单模型手动调试**

```bash
conda activate eriq-qwen
python stepbystep/engine/run_from_config.py --config stepbystep/configs/templates/qwen2.5-vl-3b.yaml
```

### 3. 查看结果

评测完成后，查看汇总报告：

```bash
cat results/smoke20_minimal_fp16_20b/summary.md
```

***

## 🔧 核心模块说明

| 模块       | 路径                                                                                                                | 作用                       |
| :------- | :---------------------------------------------------------------------------------------------------------------- | :----------------------- |
| **推理引擎** | [run\_from\_config.py](file:///e:/Projects/eriq-eval/stepbystep/engine/run_from_config.py)                        | 解析 YAML 并执行推理，处理图像/文本输入。 |
| **智能重试** | [smart\_retry\_daemon.py](file:///e:/Projects/eriq-eval/stepbystep/retry/smart_retry_daemon.py)                   | 补测机制，自动处理 CUDA 挂死和失败任务。  |
| **日志清洗** | [clean\_legacy\_logs\_pipeline.py](file:///e:/Projects/eriq-eval/stepbystep/tools/clean_legacy_logs_pipeline.py)  | 降噪引擎，将巨型日志压缩 90% 并按天分档。  |
| **系统诊断** | [run\_system\_diagnostic.sh](file:///e:/Projects/eriq-eval/stepbystep/tools/diagnostics/run_system_diagnostic.sh) | 检查显存、驱动、依赖，生成环境诊断报告。     |

***

## ⚠️ 风险提示

- **显存峰值**: 部分模型（如 InternVL 20B+）在处理高分辨率图片或多图输入时可能触发显存峰值，建议开启 `ERIQ_SAFE_ATTN=1`。
- **网络波动**: 首次运行会自动从 ModelScope 下载权重，请确保网络连接稳定或已配置 `MODELSCOPE_CACHE`。

***

*文档版本: 2026-05-08*
