# ERIQ 排查操作指南

本指南用于在真实评测机上执行全量模型系统性排查，覆盖以下范围�?
1. 数据格式兼容性（输入/输出结构、字段类型、维度一致性）
2. 张量注入流程（预处理、张量构造、设备迁移、精度转换）
3. 模型加载与初始化（权重加载、图结构验证、算子支持）
4. 推理阶段异常（显存占用、算子执行、CUDA 异常�?
***

## 1) 产物说明

- 报告生成脚本：`stepbystep/diagnostics/generate_system_diagnostic_report.py`
- 断言工具脚本：`stepbystep/diagnostics/assert_model_pipeline.py`
- 一键入口脚本：`stepbystep/diagnostics/run_system_diagnostic.sh`
- 报告输出（执行后）：
  - `stepbystep/diagnostics/system_diagnostic_report.md`
  - `stepbystep/diagnostics/system_diagnostic_report.json`

***

## 2) 执行顺序

### 2.1 先做离线系统体检（推荐先执行�?
```bash
cd /mnt/e/Projects/eriq-eval
bash stepbystep/diagnostics/run_system_diagnostic.sh
```

该步骤不跑全量推理，只分析：

- run\_list（待测模型全集）
- state（守护进程当前状态）
- 历史日志（智能重�?+ 旧补测日志）

***

### 2.2 对单模型做静态断言（不执行推理�?
```bash
python /mnt/e/Projects/eriq-eval/stepbystep/diagnostics/assert_model_pipeline.py \
  --config /mnt/e/Projects/eriq-eval/stepbystep/configs/fp16_budget/qwen2.5-vl-3b-fp16.yaml
```

断言项：

- 必需配置字段存在（`model_name/model_id/loader/inference`�?- `inference.do_sample=false`（保证控制变量）
- 关键 loader 参数存在（`trust_remote_code/torch_dtype`�?
***

### 2.3 对单模型做动态断言（limit=1，手动触发）

```bash
python /mnt/e/Projects/eriq-eval/stepbystep/diagnostics/assert_model_pipeline.py \
  --config /mnt/e/Projects/eriq-eval/stepbystep/configs/fp16_budget/internvl2.5-1b-fp16.yaml \
  --subset QA_RELATIVE_POS_GROUNDING.json \
  --output /mnt/e/Projects/eriq-eval/results/_diag_internvl2.5-1b.json \
  --execute
```

动态断言项：

- 输出文件存在且为 JSON 数组
- 记录字段完整：`id/prediction/gt_answer/raw_output`
- 预测格式合法：`A-F / Yes / No / 空串`

***

## 3) 全量模型检查清单执行模�?
�?run\_list 遍历每个模型，逐条执行静态断言�?
```bash
python - <<'PY'
import yaml, subprocess
run_list="/mnt/e/Projects/eriq-eval/stepbystep/configs/fp16_budget/run_list_fp16_budget.yaml"
with open(run_list,"r",encoding="utf-8") as f:
    runs=yaml.safe_load(f)["runs"]
for r in runs:
    cfg=r["config"]
    cmd=[
        "python",
        "/mnt/e/Projects/eriq-eval/stepbystep/diagnostics/assert_model_pipeline.py",
        "--config", cfg
    ]
    print("[check]", cfg)
    subprocess.call(cmd)
PY
```

***

## 4) 问题分布统计与根因分级口�?
报告中根因分类：

- `data_tensor_shape_mismatch` -> 优先�?`P0`
- `cuda_oom` -> 优先�?`P1`
- `model_download_or_network` -> 优先�?`P1`
- `cuda_unknown_error` -> 优先�?`P2`
- `other_runtime_error` -> 优先�?`P3`

优先处理顺序�?
1. 结构性错误（同类模型全挂�?2. 资源/下载稳定性（影响覆盖率）
3. 低频未知错误（尾部收敛）

***

## 5) 复现与记录规�?
每个失败模型至少记录�?
- 模型配置路径
- 复现命令
- 日志片段（含堆栈�?- 错误分类与优先级
- 是否已修�?待验�?
建议记录模板�?
```markdown
### <task_id>
- 状�? FAIL
- 分类: data_tensor_shape_mismatch
- 复现命令: `python ... --config ... --subset ... --limit 1`
- 栈摘�? `ValueError: wrong pixel_values size: ...`
- 修复建议: 统一 pixel_values �?[N,C,H,W]
```

