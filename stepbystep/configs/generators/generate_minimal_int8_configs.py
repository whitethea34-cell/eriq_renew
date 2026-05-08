#!/usr/bin/env python3
"""
生成“最小参�?+ INT8”专用配置（独立于已�?minimal 多量化流程）
不会覆盖已有:
- stepbystep/configs/quant
- results/smoke20_minimal
"""

import os
import copy
import yaml

PROJECT_ROOT = "/mnt/e/Projects/eriq-eval"
STEP_DIR = os.path.join(PROJECT_ROOT, "stepbystep")
OUT_DIR = os.path.join(STEP_DIR, "configs/quant")


MODEL_SPECS_INT8 = [
    {
        "series": "qwen2.5-vl",
        "size": "3b",
        "model_name": "Qwen2.5-VL-3B",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "loader": {
            "class": "Qwen2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen3-vl",
        "size": "2b",
        "model_name": "Qwen3-VL-2B-Instruct",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen3-VL-2B-Instruct",
        "loader": {
            "class": "Qwen3VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "env": "eriq-qwen3-vl-2b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen3.5-vl",
        "size": "0.8b",
        "model_name": "Qwen3.5-0.8B",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen3.5-0.8B",
        "loader": {
            "class": "Qwen3_5ForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "env": "eriq-qwen3.5-vl-0.8b",
        "run_enabled": True,
        "skip_reason": "",
        "force_fp16": True,
    },
    {
        "series": "internvl2.5",
        "size": "1b",
        "model_name": "InternVL2.5-1B",
        "model_id": "OpenGVLab/InternVL2_5-1B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL2_5-1B",
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "env": "eriq-internvl2.5-1b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "internvl3",
        "size": "1b",
        "model_name": "InternVL3-1B",
        "model_id": "OpenGVLab/InternVL3-1B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL3-1B",
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "env": "eriq-internvl3-1b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "internvl3.5",
        "size": "1b",
        "model_name": "InternVL3.5-1B",
        "model_id": "OpenGVLab/InternVL3_5-1B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL3_5-1B",
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "env": "eriq-internvl3.5-1b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "llava",
        "size": "0.5b",
        "model_name": "LLaVA-OneVision-0.5B",
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "source": "modelscope",
        "modelscope_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "loader": {
            "class": "LlavaOnevisionForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "env": "eriq-llava-0.5b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "minicpm-v",
        "size": "2.5",
        "model_name": "MiniCPM-Llama3-V-2_5",
        "model_id": "OpenBMB/MiniCPM-Llama3-V-2_5",
        "source": "modelscope",
        "modelscope_id": "OpenBMB/MiniCPM-Llama3-V-2_5",
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "env": "eriq-minicpm-v-2.5",
        "run_enabled": True,
        "skip_reason": "",
    },
]


def build_cfg(spec):
    loader = copy.deepcopy(spec["loader"])
    if spec.get("force_fp16", False):
        loader.pop("quantization", None)
    else:
        loader["quantization"] = "8bit"
    cfg = {
        "model_name": spec["model_name"],
        "model_id": spec["model_id"],
        "local_path": None,
        "source": spec["source"],
        "modelscope_id": spec["modelscope_id"],
        "modelscope_cache_dir": os.path.join(PROJECT_ROOT, "models"),
        "loader": loader,
        "inference": {
            "max_new_tokens": 32,
            "do_sample": False,
            "system_prompt": (
                "You are a helpful assistant. Answer with only the choice letter (A/B/C/D/E/F)\n"
                "for multiple choice questions, or only 'Yes'/'No' for binary questions.\n"
                "Do not explain. One character only."
            ),
        },
        "eriq": {
            "supported_modalities": ["single_image", "multi_image", "interleaved"],
            "answer_extraction_mode": "first_letter",
        },
    }
    return cfg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_list = []

    for spec in MODEL_SPECS_INT8:
        out_name = f"{spec['series']}-{spec['size']}-int8.yaml"
        out_path = os.path.join(OUT_DIR, out_name)
        cfg = build_cfg(spec)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

        run_list.append(
            {
                "config": out_path,
                "series": spec["series"],
                "size": spec["size"],
                "quant": "8bit",
                "env": spec["env"],
                "run_enabled": spec["run_enabled"],
                "skip_reason": spec["skip_reason"],
                "output_name": f"{spec['series']}-{spec['size']}-int8_smoke20.json",
            }
        )

    run_list_path = os.path.join(OUT_DIR, "run_list_minimal_int8.yaml")
    with open(run_list_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"runs": run_list}, f, allow_unicode=False, sort_keys=False)

    print(f"[OK] INT8 配置目录: {OUT_DIR}")
    print(f"[OK] INT8 run list: {run_list_path}")
    print(f"[OK] 共生�?{len(run_list)} �?INT8 配置")


if __name__ == "__main__":
    main()
