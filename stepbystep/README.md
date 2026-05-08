# ERIQ 评测核心引擎与运维工具链 (Step-by-Step)

本目录（`stepbystep/`）是整个 ERIQ 评测项目的核心，包含了从环境搭建、配置生成、模型推理到故障恢复、结果汇总的全链路逻辑。项目采用模块化设计，旨在提供工业级、可复用的模型评测 SOP。

---

## 📌 目录索引

- [1. 项目背景与目标](#1-项目背景与目标)
- [2. 整体架构](#2-整体架构)
- [3. 核心模块详解](#3-核心模块详解)
  - [3.1 推理引擎 (Engine)](#31-推理引擎-engine)
  - [3.2 配置管理 (Configs)](#32-配置管理-configs)
  - [3.3 运行脚本 (Scripts)](#33-运行脚本-scripts)
  - [3.4 智能重试 (Retry)](#34-智能重试-retry)
  - [3.5 运维与工具 (Tools)](#35-运维与工具-tools)
- [4. 目录树与关键文件说明](#4-目录树与关键文件说明)
- [5. 快速开始 (Quick Start)](#5-快速开始-quick-start)
- [6. 环境依赖与环境变量](#6-环境依赖与环境变量)
- [7. 常见错误排查 (FAQ)](#7-常见错误排查-faq)
- [8. 贡献指南](#8-贡献指南)

---

## 1. 项目背景与目标

在视觉语言模型（VLM）的具身推理评测中，常常面临模型版本碎片化、依赖冲突、显存管理困难以及长时运行不稳定等挑战。本工具链的目标是：
- **统一化**：提供统一的模型配置与运行入口。
- **自动化**：实现配置批量生成与全流程自动化跑测。
- **鲁棒性**：通过隔离环境与智能重试机制，确保大规模评测任务的顺利完成。
- **透明化**：标准化的日志与评分报告，确保评测结果可追溯、可对比。

---

## 2. 整体架构

项目围绕 **“配置驱动 (Config-Driven)”** 和 **“环境隔离 (Env-Isolation)”** 两个核心原则构建：

1.  **输入层 (Input)**: 准备 ERIQ 数据集与手工编写的模型 YAML 原型模板。
2.  **生成层 (Generation)**: 通过 `generators` 脚本，根据模板批量产出 FP16、INT8、AWQ 等不同量化的运行配置。
3.  **执行层 (Execution)**: `scripts` 脚本在独立的 Conda 环境中拉起 `engine` 推理引擎，确保依赖不冲突。
4.  **保障层 (Guarantee)**: `retry` 模块中的守护进程监控 CUDA 状态，处理挂死并自动断点补测。
5.  **输出层 (Output)**: 产出标准 JSON 预测结果、降噪日志以及多维度对比报告。

---

## 3. 核心模块详解

### 3.1 推理引擎 (Engine)

#### [run_from_config.py](file:///e:/Projects/eriq-eval/stepbystep/engine/run_from_config.py)
- **功能说明**: 核心推理入口。解析 YAML 配置，加载模型与处理器，遍历 ERIQ 数据集并输出预测。
- **输入输出**:
  - **输入**: `--config` 指向的 YAML 配置文件。
  - **输出**: `results/` 下的预测结果 JSON。
- **使用示例**:
  ```bash
  python stepbystep/engine/run_from_config.py --config stepbystep/configs/templates/qwen2.5-vl-3b.yaml
  ```

#### [run_from_config_per_subset.py](file:///e:/Projects/eriq-eval/stepbystep/engine/run_from_config_per_subset.py)
- **功能说明**: 增强型引擎。将 15 个子任务拆分为独立进程运行，每完成一个子任务释放一次显存，解决 CUDA 状态污染。
- **输入输出**: 同上，但执行粒度更细，稳定性更高。
- **使用示例**: 被 `run_smoke*.sh` 脚本内部调用。

---

### 3.2 配置管理 (Configs)

- **[templates/](file:///e:/Projects/eriq-eval/stepbystep/configs/templates/)**: 手工维护的模型原型，定义了模型 ID 和加载参数。
- **[generators/](file:///e:/Projects/eriq-eval/stepbystep/configs/generators/)**: 包含 `generate_minimal_quant_configs.py` 等脚本，用于批量生成特定实验的配置文件。
- **[run_lists/](file:///e:/Projects/eriq-eval/stepbystep/configs/run_lists/)**: 定义实验的模型执行清单，控制批量跑测的范围。

---

### 3.3 运行脚本 (Scripts)

#### [run_smoke20_minimal_fp16_20b.sh](file:///e:/Projects/eriq-eval/stepbystep/scripts/run_smoke20_minimal_fp16_20b.sh)
- **功能说明**: FP16 系列的主力批处理脚本。执行从配置生成到打分汇总的全流程。
- **输入输出**: 自动化执行，最终在 `results/` 生成 `summary.md`。
- **使用示例**: `bash stepbystep/scripts/run_smoke20_minimal_fp16_20b.sh`

#### [setup_envs.sh](file:///e:/Projects/eriq-eval/stepbystep/scripts/setup_envs.sh)
- **功能说明**: 一键初始化/修复所有模型系列所需的隔离 Conda 环境。

---

### 3.4 智能重试 (Retry)

#### [smart_retry_daemon.py](file:///e:/Projects/eriq-eval/stepbystep/retry/smart_retry_daemon.py)
- **功能说明**: 智能守护进程。监控评测进度，若发现进程挂死（CUDA 错误或超时），则自动重启。
- **使用示例**:
  ```bash
  python stepbystep/retry/smart_retry_daemon.py --config stepbystep/retry/configs/fp16_20b.yaml
  ```

---

### 3.5 运维与工具 (Tools)

#### [clean_legacy_logs_pipeline.py](file:///e:/Projects/eriq-eval/stepbystep/tools/clean_legacy_logs_pipeline.py)
- **功能说明**: 日志清洗引擎。过滤 tqdm 进度条噪音，实现“阅后即焚”式归档。
- **使用示例**: `python stepbystep/tools/clean_legacy_logs_pipeline.py logs/standard/`

#### [diagnostics/run_system_diagnostic.sh](file:///e:/Projects/eriq-eval/stepbystep/tools/diagnostics/run_system_diagnostic.sh)
- **功能说明**: 系统体检工具。一键输出 GPU 状态、驱动及依赖库对比报告。

---

## 4. 目录结构与核心脚本说明

本目录采用模块化设计，通过解耦推理引擎、配置文件与运维工具，确保评测任务的可维护性与自动化：

```text
stepbystep/
├── engine/         # 核心引擎 (run_from_config.py): 解析 YAML 配置执行推理，支持子任务隔离运行以优化显存。
├── configs/        # 配置体系 (generators/): 维护模型原型模板，支持批量产出 FP16/INT8/AWQ 多位宽运行配置。
├── scripts/        # 运行流水线 (run_smoke*.sh): 一键自动化评测脚本，涵盖推理、打分与汇总全流程。
├── retry/          # 故障恢复 (smart_retry_daemon.py): 智能守护进程，实时监控进程挂死并支持自动补测。
└── tools/          # 运维工具 (clean_legacy_logs_pipeline.py): 提供日志降噪清洗、系统诊断及文档自动维护。
```

---

## 5. 快速开始 (Quick Start)

1.  **环境准备**:
    ```bash
    # 在 WSL2 环境下，初始化 E 盘结构
    bash stepbystep/tools/setup/setup_e_drive.sh
    ```
2.  **创建隔离环境**:
    ```bash
    # 耗时较长，请确保网络通畅
    bash stepbystep/scripts/setup_envs.sh
    ```
3.  **启动批量评测**:
    ```bash
    # 以 FP16 <= 20B 模型系列为例
    bash stepbystep/scripts/run_smoke20_minimal_fp16_20b.sh
    ```

---

## 6. 环境依赖与环境变量

- **Python**: 3.12.10
- **关键环境变量**:
  - `ERIQ_SAFE_ATTN=1`: 强制使用原生 Attention，解决部分模型在 FlashAttention2 下的稳定性问题。
  - `HF_ENDPOINT=https://hf-mirror.com`: 国内镜像加速。
  - `MODELSCOPE_CACHE`: 建议指向 E 盘大容量分区。

---

## 7. 常见错误排查 (FAQ)

| 错误表现 | 可能原因 | 解决方案 |
| :--- | :--- | :--- |
| `CUDA out of memory` | 视觉输入过大或显存泄漏 | 开启 `ERIQ_SAFE_ATTN=1` 或使用 `run_from_config_per_subset.py` |
| `ImportError` | Conda 环境未正确激活 | 检查 `setup_envs.sh` 输出，确保环境已创建。 |
| `Unknown Error (Async)` | CUDA 驱动或硬件底层异常 | 运行 `retry/smart_retry_daemon.py` 进行自动断点重试。 |
| 进度条刷屏导致日志过大 | `tqdm` 默认刷新频率过高 | 运行 `tools/clean_legacy_logs_pipeline.py` 进行日志降噪。 |

---

## 8. 贡献指南

1.  **提交 Issue**: 发现 Bug 或模型兼容性问题，请附带 `logs/` 下的精简日志。
2.  **提交 PR**:
    - **新增模型**: 在 `configs/templates/` 提交 YAML 原型，并在对应 `generators` 中注册。
    - **代码风格**: 遵循 PEP8 规范，注释建议使用中文。
3.  **测试要求**: 修改核心逻辑后，请务必运行 `tools/setup/check_env.sh` 验证。

---
*最后更新: 2026-05-08*
