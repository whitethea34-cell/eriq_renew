#!/usr/bin/env python3
"""
生成 AWQ 量化版本的测试配�?适配 vLLM / TensorRT-LLM 部署，开�?enable_chunked_prefill �?KV Cache 量化
控制 max_model_len 避免长序列爆显存
"""

import copy
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "stepbystep", "configs/awq")

# AWQ 量化模型列表
MODEL_SPECS_AWQ = [
    # Qwen2.5-VL 系列 AWQ
    {
        "series": "qwen2.5-vl",
        "size": "3b-awq",
        "model_name": "Qwen2.5-VL-3B-AWQ",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "param_count": 3,
        "loader": {
            "class": "Qwen2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "deployment": {
            "framework": "vLLM",
            "enable_chunked_prefill": True,
            "kv_cache_quantization": True,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.8,
            "quantization": "awq",
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen2.5-vl",
        "size": "7b-awq",
        "model_name": "Qwen2.5-VL-7B-AWQ",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "param_count": 7,
        "loader": {
            "class": "Qwen2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "deployment": {
            "framework": "vLLM",
            "enable_chunked_prefill": True,
            "kv_cache_quantization": True,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.8,
            "quantization": "awq",
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen2.5-vl",
        "size": "32b-awq",
        "model_name": "Qwen2.5-VL-32B-AWQ",
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "param_count": 32,
        "loader": {
            "class": "Qwen2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "deployment": {
            "framework": "vLLM",
            "enable_chunked_prefill": True,
            "kv_cache_quantization": True,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "quantization": "awq",
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen2.5-vl",
        "size": "72b-awq",
        "model_name": "Qwen2.5-VL-72B-AWQ",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "param_count": 72,
        "loader": {
            "class": "Qwen2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "deployment": {
            "framework": "vLLM",
            "enable_chunked_prefill": True,
            "kv_cache_quantization": True,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "quantization": "awq",
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    # InternLM-XComposer2.5-VL 系列 AWQ
    {
        "series": "xcomposer2.5-vl",
        "size": "7b-awq",
        "model_name": "InternLM-XComposer2.5-7B-AWQ",
        "model_id": "internlm/internlm-xcomposer2d5-7b-awq",
        "source": "modelscope",
        "modelscope_id": "internlm/internlm-xcomposer2d5-7b-awq",
        "param_count": 7,
        "loader": {
            "class": "IXC2_5_VLForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "deployment": {
            "framework": "vLLM",
            "enable_chunked_prefill": True,
            "kv_cache_quantization": True,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "quantization": "awq",
        },
        "env": "eriq-xcomposer2.5-vl-7b",
        "run_enabled": True,
        "skip_reason": "",
    },
    # MiniCPM-V 系列 AWQ
    {
        "series": "minicpm-v",
        "size": "2.5-awq",
        "model_name": "MiniCPM-V-2.5-AWQ",
        "model_id": "OpenBMB/MiniCPM-V-2_5-awq",
        "source": "modelscope",
        "modelscope_id": "OpenBMB/MiniCPM-V-2_5-awq",
        "param_count": 2.5,
        "loader": {
            "class": "MiniCPMV",
            "processor_class": "AutoProcessor",
            "trust_remote_code": True,
            "torch_dtype": "float16",
            "device_map": "auto",
        },
        "deployment": {
            "framework": "vLLM",
            "enable_chunked_prefill": True,
            "kv_cache_quantization": True,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "quantization": "awq",
        },
        "env": "eriq-minicpm-v-2.5",
        "run_enabled": True,
        "skip_reason": "",
    },
]


def build_cfg(spec):
    """�?spec 转换为最终的 YAML 配置字典"""
    loader = copy.deepcopy(spec["loader"])
    loader["torch_dtype"] = "float16"
    cfg = {
        "model_name": spec["model_name"],
        "model_id": spec["model_id"],
        "local_path": None,
        "source": spec["source"],
        "modelscope_id": spec["modelscope_id"],
        "modelscope_cache_dir": os.path.join(PROJECT_ROOT, "models"),
        "loader": loader,
        "deployment": spec["deployment"],
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
    for spec in MODEL_SPECS_AWQ:
        out_name = f"{spec['series']}-{spec['size']}-awq.yaml"
        out_path = os.path.join(OUT_DIR, out_name)
        cfg = build_cfg(spec)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
        run_list.append(
            {
                "config": out_path,
                "series": spec["series"],
                "size": spec["size"],
                "quant": "awq",
                "param_count": spec["param_count"],
                "deployment": spec["deployment"],
                "env": spec["env"],
                "run_enabled": spec["run_enabled"],
                "skip_reason": spec["skip_reason"],
                "output_name": f"{spec['series']}-{spec['size']}-awq_smoke20.json",
            }
        )
    run_list_path = os.path.join(OUT_DIR, "run_list_minimal_awq.yaml")
    with open(run_list_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"runs": run_list}, f, allow_unicode=False, sort_keys=False)
    print(f"[OK] AWQ 配置目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
