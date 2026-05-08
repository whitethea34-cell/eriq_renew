#!/usr/bin/env python3
"""
生成 FP16 精度、无参数量限制的测试配置
适配 vLLM / TensorRT-LLM 部署，开�?enable_chunked_prefill �?KV Cache 量化
控制 max_model_len 避免长序列爆显存
"""

import copy
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "stepbystep", "configs/fp16")

# 无参数量限制�?FP16 模型列表
MODEL_SPECS_FP16_NO_LIMIT = [
    # Qwen 系列
    {
        "series": "qwen2.5-vl",
        "size": "3b",
        "model_name": "Qwen2.5-VL-3B-FP16",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-3B-Instruct",
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
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen2.5-vl",
        "size": "7b",
        "model_name": "Qwen2.5-VL-7B-FP16",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-7B-Instruct",
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
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen2.5-vl",
        "size": "32b",
        "model_name": "Qwen2.5-VL-32B-FP16",
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-32B-Instruct",
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
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen2.5-vl",
        "size": "72b",
        "model_name": "Qwen2.5-VL-72B-FP16",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen2.5-VL-72B-Instruct",
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
        },
        "env": "eriq-qwen2.5-vl-3b",
        "run_enabled": True,
        "skip_reason": "",
    },
    # InternVL 系列（大模型�?    {
        "series": "internvl2.5",
        "size": "26b",
        "model_name": "InternVL2.5-26B-FP16",
        "model_id": "OpenGVLab/InternVL2_5-26B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL2_5-26B",
        "param_count": 26,
        "loader": {
            "class": "InternVLChatModel",
            "processor_class": "InternVLChatProcessor",
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
        },
        "env": "eriq-internvl2.5-1b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "internvl2.5",
        "size": "38b",
        "model_name": "InternVL2.5-38B-FP16",
        "model_id": "OpenGVLab/InternVL2_5-38B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL2_5-38B",
        "param_count": 38,
        "loader": {
            "class": "InternVLChatModel",
            "processor_class": "InternVLChatProcessor",
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
        },
        "env": "eriq-internvl2.5-1b",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "internvl2.5",
        "size": "78b",
        "model_name": "InternVL2.5-78B-FP16",
        "model_id": "OpenGVLab/InternVL2_5-78B",
        "source": "modelscope",
        "modelscope_id": "OpenGVLab/InternVL2_5-78B",
        "param_count": 78,
        "loader": {
            "class": "InternVLChatModel",
            "processor_class": "InternVLChatProcessor",
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
        },
        "env": "eriq-internvl2.5-1b",
        "run_enabled": True,
        "skip_reason": "",
    },
    # Qwen3.5 系列（大模型�?    {
        "series": "qwen3.5",
        "size": "27b",
        "model_name": "Qwen3.5-27B-FP16",
        "model_id": "Qwen/Qwen3.5-27B",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen3.5-27B",
        "param_count": 27,
        "loader": {
            "class": "Qwen3ForCausalLM",
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
        },
        "env": "eriq-qwen",
        "run_enabled": True,
        "skip_reason": "",
    },
    {
        "series": "qwen3.5",
        "size": "35b-a3b",
        "model_name": "Qwen3.5-35B-A3B-FP16",
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "source": "modelscope",
        "modelscope_id": "Qwen/Qwen3.5-35B-A3B",
        "param_count": 35,
        "loader": {
            "class": "Qwen3ForCausalLM",
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
        },
        "env": "eriq-qwen",
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
    for spec in MODEL_SPECS_FP16_NO_LIMIT:
        out_name = f"{spec['series']}-{spec['size']}-fp16.yaml"
        out_path = os.path.join(OUT_DIR, out_name)
        cfg = build_cfg(spec)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
        run_list.append(
            {
                "config": out_path,
                "series": spec["series"],
                "size": spec["size"],
                "quant": "fp16",
                "param_count": spec["param_count"],
                "deployment": spec["deployment"],
                "env": spec["env"],
                "run_enabled": spec["run_enabled"],
                "skip_reason": spec["skip_reason"],
                "output_name": f"{spec['series']}-{spec['size']}-fp16_smoke20.json",
            }
        )
    run_list_path = os.path.join(OUT_DIR, "run_list_minimal_fp16_no_limit.yaml")
    with open(run_list_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"runs": run_list}, f, allow_unicode=False, sort_keys=False)
    print(f"[OK] FP16 无限制配置目�? {OUT_DIR}")


if __name__ == "__main__":
    main()
