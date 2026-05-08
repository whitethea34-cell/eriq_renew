# ERIQ 模型系统性排查报告（离线分析�?
- 生成时间: `2026-05-07T10:05:24`
- 项目根目�? `/mnt/e/Projects/eriq-eval`
- 模型总数: `8`
- 系列分布: `{'qwen2.5-vl': 1, 'qwen3-vl': 1, 'qwen3.5-vl': 1, 'internvl2.5': 1, 'internvl3': 1, 'internvl3.5': 1, 'llava': 1, 'smolvlm': 1}`
- 状态分�? `{'in_progress': 1, 'pending': 4, 'completed': 3}`

## 1) 问题分布统计

| 根因类别 | 数量 | 优先�?|
|---|---:|---|
| `other_runtime_error` | 26 | `P3` |
| `cuda_oom` | 15 | `P1` |
| `cuda_unknown_error` | 8 | `P2` |

## 2) 根因分类与修复优先级

- `P0` 数据/张量结构问题: 典型�?`wrong pixel_values size`，会导致整模型整子任务全失败�?- `P1` 资源与下载问�? `cuda_oom`、模型拉取不稳定，影响稳定性与覆盖率�?- `P2` CUDA 异步未知错误: 常由上游 CUDA 异常污染触发�?- `P3` 其他运行时错�? 需按具体栈定位�?
## 3) 栈样例（每类首条�?
- `other_runtime_error`: `1: CUDA driver error: device not ready @ /mnt/e/Projects/eriq-eval/logs/standard/retry-int8-full/smart_retry_daemon.log:33344`
- `cuda_oom`: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 GiB. GPU 0 has a total capacity of 47.99 GiB of which 0 bytes is free. Of the allocated memory 3.35 GiB is allocated by PyTorch, and 65.94 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables) @ /mnt/e/Projects/eriq-eval/logs/standard/retry-int8-full/smart_retry_daemon.log:33484`
- `cuda_unknown_error`: `CUDA error: unknown error @ /mnt/e/Projects/eriq-eval/logs/standard/retry-int8-full/smart_retry_daemon.log:35066`

## 4) 每模型检查清�?
| 模型(task_id) | 数据格式 | 张量流程 | 加载初始�?| 推理异常 | 综合状�?| 现状 |
|---|---|---|---|---|---|---|
| `qwen2.5-vl|3b|8bit` | PENDING | PENDING | PENDING | PENDING | RUNNING | `in_progress` |
| `qwen3-vl|2b|8bit` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |
| `qwen3.5-vl|0.8b|8bit` | PENDING | RISK | PENDING | FAIL | FAIL | `pending` |
| `internvl2.5|1b|8bit` | PASS | PASS | PASS | PASS | PASS | `completed` |
| `internvl3|1b|8bit` | PASS | PASS | PASS | PASS | PASS | `completed` |
| `internvl3.5|1b|8bit` | PASS | PASS | PASS | PASS | PASS | `completed` |
| `llava|0.5b|8bit` | PENDING | RISK | PENDING | FAIL | FAIL | `pending` |
| `smolvlm|512m|8bit` | PENDING | PENDING | PENDING | PENDING | PENDING | `pending` |

## 5) 复现步骤（统一�?
```bash
cd /mnt/e/Projects/eriq-eval
python stepbystep/engine/run_from_config.py \
  --config /mnt/e/Projects/eriq-eval/stepbystep/configs/fp16_budget/<model>.yaml \
  --subset-file QA_RELATIVE_POS_GROUNDING.json \
  --limit 1 \
  --output /mnt/e/Projects/eriq-eval/results/_diag_<model>.json
```

## 6) 结论

- 现阶段首要问题是 `data_tensor_shape_mismatch`，优先完成所�?InternVL 路径验证�?- 完成结构性修复后，再集中处理 `cuda_oom` �?`cuda_unknown_error`�?- 建议使用 `stepbystep/diagnostics/run_system_diagnostic.sh` 做批量可重复检查�?
