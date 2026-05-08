#!/usr/bin/env python3
"""
根据“最小参数清单”自动生成多量化 YAML（基于真实可用模型）:
- fp16: 原始权重
- 4bit: bitsandbytes 4bit（仅当模型允许）
- awq: 预量�?AWQ 权重（仅当提�?awq_modelscope_id�?"""

import os
import copy
import yaml

PROJECT_ROOT = "/mnt/e/Projects/eriq-eval"
STEP_DIR = os.path.join(PROJECT_ROOT, "stepbystep")
OUT_DIR = os.path.join(STEP_DIR, "configs/quant")


MODEL_SPECS = [
    {
        "series": "qwen2.5-vl",
        "size": "3b",
        "model_name": "Qwen2.5-VL-3B",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "awq_modelscope_id": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "supported_quants": ["fp16", "4bit", "awq"],
        "loader": {
            "class": "Qwen2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
    },
    {
        "series": "qwen3-vl",
        "size": "2b",
        "model_name": "Qwen3-VL-2B-Instruct",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen3-VL-2B-Instruct",
        "supported_quants": ["fp16"],
        "loader": {
            "class": "Qwen3VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-qwen3-vl-2b",
        "run_enabled": True,
    },
    {
        "series": "qwen3.5-vl",
        "size": "0.8b",
        "model_name": "Qwen3.5-0.8B",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen3.5-0.8B",
        "supported_quants": ["fp16"],
        "loader": {
            "class": "Qwen3_5ForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-qwen3.5-vl-0.8b",
        "run_enabled": True,
    },
    {
        "series": "internvl2.5",
        "size": "1b",
        "model_name": "InternVL2.5-1B",
        "model_id": "OpenGVLab/InternVL2_5-1B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL2_5-1B",
        "supported_quants": ["fp16"],
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-internvl2.5-1b",
        "run_enabled": True,
    },
    {
        "series": "internvl3",
        "size": "1b",
        "model_name": "InternVL3-1B",
        "model_id": "OpenGVLab/InternVL3-1B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL3-1B",
        "supported_quants": ["fp16"],
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-internvl3-1b",
        "run_enabled": True,
    },
    {
        "series": "internvl3.5",
        "size": "1b",
        "model_name": "InternVL3.5-1B",
        "model_id": "OpenGVLab/InternVL3_5-1B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL3_5-1B",
        "supported_quants": ["fp16"],
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-internvl3.5-1b",
        "run_enabled": True,
    },
    {
        "series": "llava",
        "size": "0.5b",
        "model_name": "LLaVA-OneVision-0.5B",
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "source": None,
        "modelscope_id": None,
        "supported_quants": ["fp16"],
        "loader": {
            "class": "LlavaOnevisionForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-llava-0.5b",
        "run_enabled": False,
        "run_note": "当前通用分支未完�?LLaVA OneVision 专用 prompt 对齐，默认不�?,
    },
    {
        "series": "minicpm-v",
        "size": "2.5",
        "model_name": "MiniCPM-Llama3-V-2_5",
        "model_id": "OpenBMB/MiniCPM-Llama3-V-2_5",
        "source": "modelscope",
        "modelscope_id": "OpenBMB/MiniCPM-Llama3-V-2_5",
        "supported_quants": ["fp16"],
        "loader": {
            "class": "AutoModel",
            "processor_class": "AutoTokenizer",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "processor_id": None,
        "env": "eriq-minicpm-v-2.5",
        "run_enabled": False,
        "run_note": "MiniCPM 需要单独模板适配，默认不�?,
    },
]


def build_base_cfg(spec):
    cfg = {
        "model_name": spec["model_name"],
        "model_id": spec["model_id"],
        "local_path": None,
        "loader": copy.deepcopy(spec["loader"]),
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
    if spec.get("source"):
        cfg["source"] = spec["source"]
    if spec.get("modelscope_id"):
        cfg["modelscope_id"] = spec["modelscope_id"]
        cfg["modelscope_cache_dir"] = os.path.join(PROJECT_ROOT, "models")
    if spec.get("processor_id"):
        cfg["processor_id"] = spec["processor_id"]
    return cfg


def apply_quant(cfg, quant, spec):
    out = copy.deepcopy(cfg)
    if quant == "fp16":
        out["loader"]["quantization"] = None
        out["loader"]["torch_dtype"] = "float16"
    elif quant == "4bit":
        out["loader"]["quantization"] = "4bit"
        out["loader"]["torch_dtype"] = "float16"
    elif quant == "awq":
        awq_modelscope_id = spec.get("awq_modelscope_id")
        awq_model_id = spec.get("awq_model_id")
        if not awq_modelscope_id and not awq_model_id:
            raise ValueError(f"{spec['series']}-{spec['size']} does not provide awq model id")
        out["loader"]["quantization"] = "awq"
        out["loader"]["torch_dtype"] = "float16"
        if awq_model_id:
            out["awq_model_id"] = awq_model_id
        if awq_modelscope_id:
            out["awq_modelscope_id"] = awq_modelscope_id
            out["awq_modelscope_cache_dir"] = out.get(
                "modelscope_cache_dir",
                os.path.join(PROJECT_ROOT, "models")
            )
        out["processor_id"] = out.get("processor_id") or out["model_id"]
    else:
        raise ValueError(f"unsupported quant: {quant}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    quant_modes = ["fp16", "4bit", "awq"]
    run_list = []

    for spec in MODEL_SPECS:
        base_cfg = build_base_cfg(spec)
        supported = set(spec.get("supported_quants", ["fp16"]))
        for quant in quant_modes:
            if quant not in supported:
                run_list.append(
                    {
                        "config": None,
                        "series": spec["series"],
                        "size": spec["size"],
                        "quant": quant,
                        "env": spec["env"],
                        "run_enabled": False,
                        "skip_reason": f"unsupported_quant:{quant}",
                        "output_name": f"{spec['series']}-{spec['size']}-{quant}_smoke20.json",
                    }
                )
                continue

            cfg = apply_quant(base_cfg, quant, spec)
            file_name = f"{spec['series']}-{spec['size']}-{quant}.yaml"
            out_path = os.path.join(OUT_DIR, file_name)
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)

            run_list.append(
                {
                    "config": out_path,
                    "series": spec["series"],
                    "size": spec["size"],
                    "quant": quant,
                    "env": spec["env"],
                    "run_enabled": spec["run_enabled"],
                    "skip_reason": spec.get("run_note", ""),
                    "output_name": f"{spec['series']}-{spec['size']}-{quant}_smoke20.json",
                }
            )

    run_list_path = os.path.join(OUT_DIR, "run_list_minimal.yaml")
    with open(run_list_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"runs": run_list}, f, allow_unicode=False, sort_keys=False)

    print(f"[OK] 生成配置目录: {OUT_DIR}")
    print(f"[OK] 生成 run list: {run_list_path}")
    print(f"[OK] 共生�?{len(run_list)} 份量化配�?)


if __name__ == "__main__":
    main()
