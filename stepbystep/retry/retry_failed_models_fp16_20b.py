import os
import shlex
import subprocess
import yaml
import pathlib

project_root = "/mnt/e/Projects/eriq-eval"
step_dir = f"{project_root}/stepbystep"
results_dir = f"{project_root}/results/smoke20_minimal_fp16_20b"
run_list = yaml.safe_load(
    pathlib.Path(f"{step_dir}/configs/run_lists/run_list_fp16_budget.yaml").read_text(encoding="utf-8")
)["runs"]
runner = f"{step_dir}/run_from_config_per_subset.py"
eval_script = f"{project_root}/eval_code/eval_hf.py"

targets = {
    ("qwen2.5-vl","7b"), ("qwen2-vl","2b"), ("qwen2-vl","7b"),
    ("qwen3-vl","2b"), ("qwen3-vl","4b"), ("qwen3-vl","8b"),
    ("qwen3.5","0.8b"), ("qwen3.5","2b"), ("qwen3.5","4b"), ("qwen3.5","9b"),
    ("internvl2.5","1b"), ("internvl2.5","2b"), ("internvl2.5","8b"),
    ("internvl3","2b"), ("internvl3","8b"),
    ("internvl3.5","14b"), ("internvl3.5","20b-a3b"),
    ("minicpm-v","2.5"), ("minicpm-v","4.5"), ("minicpm-v","o-4.5"),
    ("smolvlm","2b"), ("phi-3-vision","4.2b"), ("phi-3.5-vision","4.2b"),
    ("molmo","7b-d"), ("llava","7b"),
}

for item in run_list:
    if (item["series"], str(item["size"])) not in targets:
        continue

    env = item["env"]
    cfg = item["config"]
    out = os.path.join(results_dir, item["output_name"])
    score = out.replace(".json", "_score.txt")

    print(f"[RUN] {item['series']}-{item['size']} env={env}", flush=True)

    infer_cmd = (
        f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {shlex.quote(env)} && "
        f"export ERIQ_SAFE_ATTN=${{ERIQ_SAFE_ATTN:-1}} && "
        f"export PYTORCH_CUDA_ALLOC_CONF=${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}} && "
        f"export ERIQ_SAMPLE_EMPTY_CACHE=${{ERIQ_SAMPLE_EMPTY_CACHE:-1}} && "
        f"export ERIQ_GC_EACH_SAMPLE=${{ERIQ_GC_EACH_SAMPLE:-1}} && "
        f"export ERIQ_SUBSET_COOLDOWN_SEC=${{ERIQ_SUBSET_COOLDOWN_SEC:-3}} && "
        f"export CUDA_LAUNCH_BLOCKING=${{CUDA_LAUNCH_BLOCKING:-0}} && "
        f"python {shlex.quote(runner)} --config {shlex.quote(cfg)} --limit 20 "
        f"--data-dir {shlex.quote(project_root + '/data')} --output {shlex.quote(out)}"
    )
    rc = subprocess.call(["bash", "-lc", infer_cmd])
    if rc != 0:
        print(f"[FAIL] inference: {cfg}", flush=True)
        continue

    eval_cmd = (
        f"source \"$(conda info --base)/etc/profile.d/conda.sh\" && "
        f"conda activate {shlex.quote(env)} && "
        f"python {shlex.quote(eval_script)} {shlex.quote(out)} > {shlex.quote(score)} 2>&1"
    )
    rc = subprocess.call(["bash", "-lc", eval_cmd])
    if rc != 0:
        print(f"[WARN] eval failed: {out}", flush=True)
    else:
        print(f"[DONE] {os.path.basename(out)}", flush=True)
