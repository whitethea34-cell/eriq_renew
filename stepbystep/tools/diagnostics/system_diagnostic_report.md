# ERIQ 模型系统性排查报告（离线分析）

- 生成时间: `2026-04-30T09:15:50`
- 项目根目录: `/mnt/e/Projects/eriq-eval`
- 模型总数: `36`
- 系列分布: `{'qwen2.5-vl': 2, 'qwen2-vl': 2, 'qwen3-vl': 3, 'qwen3.5': 4, 'internvl2.5': 4, 'internvl3': 3, 'internvl3.5': 6, 'minicpm-v': 4, 'smolvlm': 3, 'phi-3-vision': 1, 'phi-3.5-vision': 1, 'molmo': 1, 'llava': 2}`
- 状态分布: `{'unknown': 13, 'pending': 19, 'completed': 2, 'in_progress': 2}`

## 1) 问题分布统计

| 根因类别 | 数量 | 优先级 |
|---|---:|---|
| `data_tensor_shape_mismatch` | 320 | `P0` |
| `cuda_oom` | 40 | `P1` |
| `cuda_unknown_error` | 3 | `P2` |

## 2) 根因分类与修复优先级

- `P0` 数据/张量结构问题: 典型为 `wrong pixel_values size`，会导致整模型整子任务全失败。
- `P1` 资源与下载问题: `cuda_oom`、模型拉取不稳定，影响稳定性与覆盖率。
- `P2` CUDA 异步未知错误: 常由上游 CUDA 异常污染触发。
- `P3` 其他运行时错误: 需按具体栈定位。

## 3) 栈样例（每类首条）

- `data_tensor_shape_mismatch`: `1: wrong pixel_values size: torch.Size([1, 6, 3, 448, 448]) @ /mnt/e/Projects/eriq-eval/logs/smart_retry/smart_retry_daemon.log:459`
- `cuda_oom`: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 245.98 GiB. GPU 0 has a total capacity of 47.99 GiB of which 27.60 GiB is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 18.66 GiB is allocated by PyTorch, and 21.22 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) @ /mnt/e/Projects/eriq-eval/logs/retry_failed_models_fp16_20b.log:1097`
- `cuda_unknown_error`: `CUDA error: unknown error @ /mnt/e/Projects/eriq-eval/logs/retry_failed_models_fp16_20b.log:10107`

## 4) 每模型检查清单

| 模型(task_id) | 数据格式 | 张量流程 | 加载初始化 | 推理异常 | 综合状态 | 现状 |
|---|---|---|---|---|---|---|
| `qwen2.5-vl|3b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `qwen2.5-vl|7b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen2-vl|2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen2-vl|7b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3-vl|2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3-vl|4b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3-vl|8b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3.5|0.8b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3.5|2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3.5|4b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3.5|9b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `internvl2.5|1b|fp16` | PASS | PASS | PASS | PASS | PASS | `completed` |
| `internvl2.5|2b|fp16` | PASS | PASS | PASS | PASS | PASS | `completed` |
| `internvl2.5|4b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl2.5|8b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3|1b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3|2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3|8b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `internvl3.5|1b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3.5|2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3.5|4b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3.5|8b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `internvl3.5|14b|fp16` | PENDING | PENDING | PENDING | PENDING | RUNNING | `in_progress` |
| `internvl3.5|20b-a3b|fp16` | PENDING | PENDING | PENDING | PENDING | RUNNING | `in_progress` |
| `minicpm-v|2.5|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `minicpm-v|2.6|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `minicpm-v|4.5|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `minicpm-v|o-4.5|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `smolvlm|256m|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `smolvlm|500m|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `smolvlm|2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `phi-3-vision|4.2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `phi-3.5-vision|4.2b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `molmo|7b-d|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `llava|0.5b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `unknown` |
| `llava|7b|fp16` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |

## 5) 复现步骤（统一）

```bash
cd /mnt/e/Projects/eriq-eval
python stepbystep/run_from_config.py \
  --config /mnt/e/Projects/eriq-eval/stepbystep/auto_configs_minimal_fp16_20b/<model>.yaml \
  --subset-file QA_RELATIVE_POS_GROUNDING.json \
  --limit 1 \
  --output /mnt/e/Projects/eriq-eval/results/_diag_<model>.json
```

## 6) 结论

- 现阶段首要问题是 `data_tensor_shape_mismatch`，优先完成所有 InternVL 路径验证。
- 完成结构性修复后，再集中处理 `cuda_oom` 与 `cuda_unknown_error`。
- 建议使用 `stepbystep/diagnostics/run_system_diagnostic.sh` 做批量可重复检查。

