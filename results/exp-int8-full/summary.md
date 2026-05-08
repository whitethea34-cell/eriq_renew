# Smoke20 计时与质量汇总

| 模型系列 | 参数 | 量化 | 环境 | 已执行 | 条数 | Accuracy | 有效率 | 错误数 | 平均时延(s) | P50(s) | P95(s) | 吞吐(item/s) | 通过 | 备注 |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen2.5-vl | 3b | 8bit | eriq-qwen2.5-vl-3b | Y | 6052 | 57.70% | 99.9% | 5 | 1.868 | 0.751 | 7.273 | 0.54 | FAIL | failed_rule |
| qwen3-vl | 2b | 8bit | eriq-qwen3-vl-2b | Y | 6052 | 64.39% | 100.0% | 0 | 0.853 | 0.418 | 3.460 | 1.17 | PASS | ok |
| qwen3.5-vl | 0.8b | 8bit | eriq-qwen3.5-vl-0.8b | Y | 6052 | 59.98% | 100.0% | 0 | 0.775 | 0.346 | 3.220 | 1.29 | PASS | ok |
| internvl2.5 | 1b | 8bit | eriq-internvl2.5-1b | Y | 6052 | 53.04% | 100.0% | 0 | 0.411 | 0.363 | 0.598 | 2.44 | FAIL | temporarily disabled for targeted rerun |
| internvl3 | 1b | 8bit | eriq-internvl3-1b | Y | 6052 | 54.30% | 100.0% | 0 | 0.597 | 0.430 | 1.845 | 1.68 | FAIL | temporarily disabled for targeted rerun |
| internvl3.5 | 1b | 8bit | eriq-internvl3.5-1b | Y | 6052 | 55.22% | 99.7% | 0 | 0.945 | 0.589 | 3.117 | 1.06 | FAIL | temporarily disabled for targeted rerun |
| smolvlm | 512m | 8bit | eriq-smolvlm | Y | 6052 | 37.82% | 94.2% | 0 | 1.002 | 0.712 | 2.844 | 1.00 | PASS | ok |
