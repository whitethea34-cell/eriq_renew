#!/usr/bin/env python3
"""
ERIQ 配置驱动的通用推理脚本
支持通过 YAML 配置文件切换不同模型，无需修改代码

用法:
    # 使用配置文件运行（推荐）
    python run_from_config.py --config ../configs/templates/qwen2.5-vl-7b.yaml
    python run_from_config.py --config ../configs/templates/internvl2.5-8b.yaml --limit 100

    # 指定数据目录和输出路�?    python run_from_config.py --config configs/templates/qwen2.5-vl-3b.yaml \
        --data-dir ~/eriq/data --output ~/eriq/results/my_run.json
"""

import argparse
import yaml
import json
import os
import sys
import glob
import re
import importlib
import copy
import time
import statistics
import gc
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from PIL import Image
import torch
import tempfile


def load_config(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dynamic_import(class_path: str):
    """
    动态导�?transformers 中的�?    例如 'transformers.AutoModel' �?'AutoProcessor'
    """
    if '.' not in class_path:
        # 默认�?transformers 导入
        module = importlib.import_module('transformers')
        return getattr(module, class_path)
    else:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


def configure_cuda_attention_backend():
    """
    可选稳定性开关：
    ERIQ_SAFE_ATTN=1 时关�?flash/mem-efficient SDP，使�?math SDP�?    这通常更慢，但能规避部分模型的 CUDA illegal memory access�?    """
    if not torch.cuda.is_available():
        return
    if os.getenv("ERIQ_SAFE_ATTN", "0") != "1":
        return
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[环境] 已启�?ERIQ_SAFE_ATTN=1（禁�?flash/mem-efficient SDP�?)
    except Exception as exc:
        print(f"[警告] 配置 CUDA 注意力后端失�? {exc}")


def is_cuda_runtime_error(err_msg: str) -> bool:
    s = (err_msg or "").lower()
    keys = [
        "cuda error",
        "out of memory",
        "illegal memory access",
        "cudacachingallocator.cpp",
        "internal assert failed",
    ]
    return any(k in s for k in keys)


def safe_cuda_cleanup(sample_level: bool = False):
    """
    �?CUDA 清理做保护，避免�?CUDA 上下文异常时再次触发致命错误�?    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        if sample_level:
            print(f"[警告] sample�?synchronize 失败（忽略）: {exc}")
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        if sample_level:
            print(f"[警告] sample�?empty_cache 失败（忽略）: {exc}")


def atomic_write_json(path: str, payload: Any) -> None:
    """
    原子�?JSON，避免进程中断时生成损坏文件�?    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def reload_model_after_cuda_error(cfg: Dict, model, processor):
    """
    发生 CUDA 运行时异常后，尝试硬释放并重载模型，清理被污染的上下文状态�?    """
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    gc.collect()
    safe_cuda_cleanup(sample_level=True)
    model, processor = load_model_and_processor(cfg)
    return model, processor


def _extract_yes_no(text: str) -> str:
    """严格提取 Yes/No，避免把噪声文本当有效答案�?""
    text = text.strip()
    if not text:
        return ""

    # 优先匹配开�?    m = re.search(r'^\s*(yes|no)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    # 次优：匹配完整单�?    m = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return ""


def _extract_mc_choice(text: str) -> str:
    """
    严格提取 A-F 选项，避免从 URL/噪声中误抽取字母�?    与官�?deterministic MC 评测口径对齐：只接受合法选项�?    """
    text = text.strip()
    if not text:
        return ""

    patterns = [
        r'^\s*(?:answer|答案)?\s*[:：是]?\s*\(?([A-F])\)?(?:[\.\):\s]|$)',
        r'(?:answer|答案|option|choice|选项)\s*[:：是]?\s*\(?([A-F])\)?',
        r'\b([A-F])\b',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def extract_answer(text: str, is_yes_no: bool = False) -> str:
    """
    官方对齐的答案标准化�?    - 多选题：仅返回 A-F
    - 判断题：仅返�?Yes/No
    其余返回空串，交给评测脚本记为无效预测�?    """
    if is_yes_no:
        return _extract_yes_no(text)
    return _extract_mc_choice(text)


def resolve_image_path(raw_path: str, data_dir: str, json_file: str) -> Optional[str]:
    """更稳健地解析图片路径，兼容相对路径、反斜杠和包�?images/ 前缀的写法�?""
    if not isinstance(raw_path, str):
        return None
    norm = raw_path.strip().replace("\\", "/")
    if not norm:
        return None

    data_dir = os.path.abspath(data_dir)
    json_dir = os.path.dirname(os.path.abspath(json_file))
    candidates = []

    if os.path.isabs(norm):
        candidates.append(norm)
    candidates.append(os.path.join(data_dir, norm))
    candidates.append(os.path.join(data_dir, "images", norm))
    candidates.append(os.path.join(json_dir, norm))

    if "/images/" in norm:
        tail = norm.split("/images/", 1)[1]
        candidates.append(os.path.join(data_dir, "images", tail))
        candidates.append(os.path.join(json_dir, "images", tail))
    elif norm.startswith("images/"):
        tail = norm[len("images/"):]
        candidates.append(os.path.join(data_dir, "images", tail))
        candidates.append(os.path.join(json_dir, "images", tail))

    seen = set()
    for c in candidates:
        if not c:
            continue
        c = os.path.abspath(os.path.normpath(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    return None


def build_messages_for_model(cfg: Dict, question_text: str, image_count: int = 0) -> List[Dict]:
    """
    根据不同模型构�?messages 格式
    这是模型切换时最常改动的部分
    """
    system = cfg['inference']['system_prompt']
    model_name = cfg['model_name'].lower()

    # 清理问题文本中的 <image> 占位符（不同模型有不同处理方式）
    cleaned_question = question_text.replace("<image>", "").strip()

    if 'llava' in model_name:
        # LLaVA-OneVision: 使用多模�?chat content 列表
        content = []
        for _ in range(max(0, image_count)):
            content.append({"type": "image"})
        content.append({"type": "text", "text": cleaned_question})
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content}
        ]

    elif 'minicpm' in model_name:
        # MiniCPM: chat 接口通常读取 user 文本并单独传 image
        prompt = f"{cleaned_question}\nAnswer with only the choice letter or Yes/No."
        return [
            {"role": "user", "content": prompt}
        ]

    elif 'qwen' in model_name:
        # Qwen 格式：content �?list，包�?image/text dict
        content = []
        # Qwen �?processor 会将 image token 与传入图片一一对应
        for _ in range(max(0, image_count)):
            content.append({"type": "image"})
        content.append({"type": "text", "text": cleaned_question})

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content}
        ]

    elif 'smolvlm' in model_name or 'phi-3' in model_name or 'molmo' in model_name:
        # 通用 HF 多模态模型：优先使用 chat template，必要时回退�?text+images
        content = []
        for _ in range(max(0, image_count)):
            content.append({"type": "image"})
        content.append(
            {
                "type": "text",
                "text": f"{cleaned_question}\nAnswer with only the choice letter or Yes/No."
            }
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content}
        ]

    elif 'internvl' in model_name:
        # InternVL 格式：使�?<image> 占位符拼接在文本�?        if image_count > 0:
            image_tokens = "\n".join(["<image>"] * image_count)
            prompt = f"{image_tokens}\n{cleaned_question}\nAnswer with only the choice letter or Yes/No."
        else:
            prompt = f"{cleaned_question}\nAnswer with only the choice letter or Yes/No."
        return [
            {"role": "user", "content": prompt}
            # 注意：InternVL �?system prompt 通常通过 generation_config 或外部传�?        ]

    else:
        # 通用格式：纯文本拼接（最保守，兼容性最好）
        prompt = f"{cleaned_question}\nAnswer with only the choice letter or Yes/No."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]


def flatten_message_content(content: Any) -> str:
    """将字符串或多模�?content 列表统一转成文本 prompt�?""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "image":
                parts.append("<image>")
            elif item_type == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def build_multimodal_inputs(processor, messages: List[Dict], images: List[Image.Image]):
    """
    为通用 HF 多模态模型构造输�?
    1) 优先使用 apply_chat_template
    2) 失败时回退�?text + images
    """
    prompt_text = flatten_message_content(messages[-1]["content"]) if messages else ""

    if hasattr(processor, "apply_chat_template"):
        try:
            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return processor(
                text=[chat_text],
                images=images if images else None,
                return_tensors="pt",
                padding=True
            )
        except Exception as exc:
            print(f"[警告] apply_chat_template 失败，回退�?text+images: {exc}")

    return processor(
        text=prompt_text,
        images=images if images else None,
        return_tensors="pt",
        padding=True
    )


def build_resized_images(images: List[Image.Image], max_side: int) -> List[Image.Image]:
    """按最长边等比缩放图片，降低视�?token 数，缓解 OOM�?""
    resized = []
    for img in images:
        w, h = img.size
        if max(w, h) <= max_side:
            resized.append(img.copy())
            continue
        scale = float(max_side) / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized.append(img.resize((new_w, new_h), Image.BICUBIC))
    return resized


def get_env_int(name: str, default: int = 0) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def normalize_internvl_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    InternVL 视觉编码器期�?4 维输�? [N, C, H, W]�?    某些处理分支可能生成 [1, N, C, H, W]，这里做统一归一化�?    """
    if not isinstance(pixel_values, torch.Tensor):
        raise TypeError(f"pixel_values must be Tensor, got {type(pixel_values)}")
    if pixel_values.dim() == 5 and pixel_values.size(0) == 1:
        pixel_values = pixel_values.squeeze(0)
    return pixel_values


def load_model_and_processor(cfg: Dict):
    """根据配置动态加载模型和处理�?""
    loader_cfg = cfg['loader']
    quant_mode = (loader_cfg.get("quantization") or "").lower().strip()
    if quant_mode in {"8bit", "4bit"} and not torch.cuda.is_available():
        print(f"[警告] 当前�?CUDA，{quant_mode} 量化不可用，自动回退�?fp16�?)
        quant_mode = ""
    model_id = resolve_model_path(cfg)
    if quant_mode == "awq":
        model_id = resolve_awq_model_path(cfg, default_model_id=model_id)

    print(f"[加载] 模型: {model_id}")
    print(f"[加载] 使用模型�? {loader_cfg['class']}")
    print(f"[加载] 使用处理器类: {loader_cfg['processor_class']}")

    # 动态导入类
    ModelClass = dynamic_import(loader_cfg['class'])
    ProcessorClass = dynamic_import(loader_cfg['processor_class'])
    # 兼容部分 trust_remote_code 老模型：�?transformers 期望存在 all_tied_weights_keys
    if not hasattr(ModelClass, "all_tied_weights_keys"):
        setattr(ModelClass, "all_tied_weights_keys", {})

    # 构造加载参�?    load_kwargs = {
        "trust_remote_code": loader_cfg.get('trust_remote_code', False),
    }

    attn_impl = loader_cfg.get("attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    if loader_cfg.get('device_map'):
        device_map = loader_cfg['device_map']
        if device_map == "auto" and not torch.cuda.is_available():
            device_map = None
        if device_map is not None:
            load_kwargs['device_map'] = device_map

    # 数据类型
    dtype_str = loader_cfg.get('torch_dtype', 'float32')
    if dtype_str == 'bfloat16':
        load_kwargs['torch_dtype'] = torch.bfloat16
    elif dtype_str == 'float16':
        load_kwargs['torch_dtype'] = torch.float16

    # 量化配置
    if quant_mode == '8bit':
        from transformers import BitsAndBytesConfig
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs['device_map'] = 'auto'
    elif quant_mode == '4bit':
        from transformers import BitsAndBytesConfig
        load_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
        load_kwargs['device_map'] = 'auto'
    elif quant_mode == 'awq':
        # AWQ 通常使用预量化权重，建议 fp16 + auto device_map
        load_kwargs['device_map'] = loader_cfg.get('device_map', 'auto')
        load_kwargs['torch_dtype'] = torch.float16

    model = ModelClass.from_pretrained(model_id, **load_kwargs)
    model.eval()

    processor_id = cfg.get("processor_id") or model_id
    processor = ProcessorClass.from_pretrained(
        processor_id,
        trust_remote_code=loader_cfg.get('trust_remote_code', False)
    )

    return model, processor


def resolve_model_path(cfg: Dict) -> str:
    """
    解析模型加载路径:
    1) local_path（本地路径）优先
    2) source=modelscope 或提�?modelscope_id 时，从魔搭下载并返回本地目录
    3) 默认使用 model_id（HuggingFace/本地路径�?    """
    local_path = cfg.get("local_path")
    if local_path:
        return os.path.expanduser(local_path)

    source = (cfg.get("source") or "").lower().strip()
    ms_model_id = cfg.get("modelscope_id")
    if source == "modelscope" or ms_model_id:
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("[错误] 配置要求使用 ModelScope，但未安�?modelscope�?)
            print("      请运�? pip install modelscope")
            sys.exit(1)

        target_model_id = ms_model_id or cfg["model_id"]
        cache_dir = cfg.get("modelscope_cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        print(f"[加载] 使用 ModelScope: {target_model_id}")
        local_dir = snapshot_download(target_model_id, cache_dir=cache_dir)
        print(f"[加载] ModelScope 本地目录: {local_dir}")
        return local_dir

    return cfg["model_id"]


def resolve_awq_model_path(cfg: Dict, default_model_id: str) -> str:
    """
    解析 AWQ 模型路径（优先级从高到低�?
    1) awq_local_path
    2) awq_modelscope_id（自动下载）
    3) awq_model_id（HF/本地可访问路径）
    4) default_model_id（兜底）
    """
    awq_local_path = cfg.get("awq_local_path")
    if awq_local_path:
        return os.path.expanduser(awq_local_path)

    awq_modelscope_id = cfg.get("awq_modelscope_id")
    if awq_modelscope_id:
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("[错误] 配置�?awq_modelscope_id，但未安�?modelscope�?)
            print("      请运�? pip install modelscope")
            sys.exit(1)

        cache_dir = cfg.get("awq_modelscope_cache_dir") or cfg.get("modelscope_cache_dir")
        if cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

        print(f"[加载] 使用 ModelScope(AWQ): {awq_modelscope_id}")
        local_dir = snapshot_download(awq_modelscope_id, cache_dir=cache_dir)
        print(f"[加载] AWQ 本地目录: {local_dir}")
        return local_dir

    awq_model_id = cfg.get("awq_model_id")
    if awq_model_id:
        return awq_model_id

    print("[警告] quantization=awq 但未提供 awq_* 路径，回退到默�?model_id�?)
    return default_model_id


def infer_with_model(cfg: Dict, model, processor, messages: List[Dict], images: List[Image.Image], max_new_tokens: int) -> str:
    """
    模型特化推理逻辑
    新增模型时，主要在这里添加分�?    """
    model_name = cfg['model_name'].lower()
    device = next(model.parameters()).device

    if 'llava' in model_name:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text

    elif 'minicpm' in model_name:
        question_text = messages[-1]["content"] if messages else ""
        if not images:
            raise RuntimeError("No valid images loaded for MiniCPM input.")
        if hasattr(model, 'chat'):
            # MiniCPM 不同版本 chat 签名略有差异，按常见顺序尝试
            try:
                response = model.chat(
                    image=images[0] if images else None,
                    msgs=[{"role": "user", "content": question_text}],
                    tokenizer=processor,
                    sampling=False,
                    max_new_tokens=max_new_tokens,
                )
            except Exception:
                generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
                try:
                    response = model.chat(
                        processor,
                        images[0] if images else None,
                        question_text,
                        generation_config,
                    )
                except Exception:
                    response = None

            if isinstance(response, tuple):
                return str(response[0])
            if response is not None:
                return str(response)

        # fallback: 用通用 processor/generate（部分兼�?HF 风格封装�?        if hasattr(processor, "apply_chat_template"):
            chat_text = processor.apply_chat_template(
                [{"role": "user", "content": question_text}],
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = processor(
                text=[chat_text],
                images=images if images else None,
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = processor(question_text, return_tensors="pt", padding=True)

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        output_text = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return output_text

    elif 'qwen' in model_name:
        # Qwen 系列推理逻辑 (Qwen2.5-VL, Qwen3-VL)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True
        )

        # 将输入移动到模型所在设�?        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 只取新生成的 token
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text

    elif 'smolvlm' in model_name or 'phi-3' in model_name or 'molmo' in model_name:
        inputs = build_multimodal_inputs(processor, messages, images)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_ids = inputs.get('input_ids')
        if input_ids is not None:
            generated = outputs[0][input_ids.shape[1]:]
        else:
            generated = outputs[0]
        return processor.decode(generated, skip_special_tokens=True)

    elif 'internvl' in model_name:
        # InternVL 系列推理逻辑
        if hasattr(model, 'chat'):
            # InternVL 原生 chat 接口
            pixel_values = None
            if images and hasattr(processor, 'image_processor'):
                pixel_values = processor.image_processor(images, return_tensors="pt").pixel_values
            elif images:
                # 备用方案：手动处理图�?                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = torch.stack([transform(img) for img in images])
            if pixel_values is not None:
                model_dtype = next(model.parameters()).dtype
                pixel_values = normalize_internvl_pixel_values(pixel_values).to(
                    device=device, dtype=model_dtype
                )

            question_text = messages[0]["content"] if messages else ""
            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            response, _ = model.chat(
                processor,
                pixel_values,
                question_text,
                generation_config,
                history=None,
                return_history=True
            )
            return response
        else:
            # 通用 fallback（用于不支持 chat 接口的变体）
            prompt = messages[0]["content"] if messages else ""
            inputs = processor(prompt, return_tensors="pt", padding=True)
            if images and hasattr(processor, 'image_processor'):
                pixel_values = processor.image_processor(images, return_tensors="pt").pixel_values
                inputs["pixel_values"] = pixel_values.to(device)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            output_text = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return output_text

    else:
        # 通用推理（最保守方式，适用于大多数 HF 模型�?        prompt = flatten_message_content(messages[-1]["content"]) if messages else ""

        if images and hasattr(processor, 'image_processor'):
            inputs = build_multimodal_inputs(processor, messages, images)
        else:
            inputs = processor(prompt, return_tensors="pt", padding=True)

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        output_text = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return output_text


def run_inference(
    cfg: Dict,
    data_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    subset_file: Optional[str] = None,
):
    """主推理循�?""
    run_start = time.perf_counter()
    quant_fallback_done = False
    try:
        model, processor = load_model_and_processor(cfg)
    except Exception as e:
        err_msg = str(e)
        quant_mode = (cfg.get("loader", {}).get("quantization") or "").lower().strip()
        if quant_mode in {"8bit", "4bit"} and (
            "has no attribute 'CB'" in err_msg
            or "all_tied_weights_keys" in err_msg
        ):
            print(
                f"[警告] 检测到加载阶段 {quant_mode} 量化兼容性错�? {err_msg}\n"
                f"      自动回退�?fp16 并重试加载�?
            )
            cfg_fp16 = copy.deepcopy(cfg)
            cfg_fp16["loader"].pop("quantization", None)
            cfg = cfg_fp16
            model, processor = load_model_and_processor(cfg)
            quant_fallback_done = True
        else:
            raise

    print("[推理] 模型加载完成�?)
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"[推理] 显存占用: {mem_gb:.2f} GB")

    # 收集数据
    json_files = sorted(glob.glob(os.path.join(data_dir, "QA_*.json")))
    if subset_file:
        subset_file = subset_file.strip()
        subset_basename = os.path.basename(subset_file)
        if not subset_basename.endswith(".json"):
            subset_basename = f"{subset_basename}.json"
        target = os.path.join(data_dir, subset_basename)
        if os.path.isfile(subset_file):
            json_files = [os.path.abspath(subset_file)]
        elif os.path.isfile(target):
            json_files = [target]
        else:
            print(f"[错误] 指定子任务文件不存在: {subset_file}")
            sys.exit(1)
    if not json_files:
        print(f"[错误] �?{data_dir} 中未找到 QA_*.json 文件�?)
        sys.exit(1)

    print(f"[推理] 发现 {len(json_files)} 个评测文�?)

    all_predictions = []
    total_attempted = 0
    total_success = 0
    total_error = 0
    inference_latencies = []
    partial_every = get_env_int("ERIQ_PARTIAL_SAVE_EVERY", 50)
    partial_path = output_path + ".partial.json"

    def save_partial_checkpoint(force: bool = False) -> None:
        if partial_every <= 0 and not force:
            return
        if total_attempted == 0:
            return
        if (not force) and (total_attempted % partial_every != 0):
            return
        payload = {
            "meta": {
                "partial": True,
                "attempted": total_attempted,
                "success": total_success,
                "error": total_error,
                "output_path": output_path,
            },
            "predictions": all_predictions,
        }
        atomic_write_json(partial_path, payload)
        print(f"\n[检查点] 已保存增量结�? {partial_path} (attempted={total_attempted})")

    for json_file in json_files:
        subset = os.path.basename(json_file).replace(".json", "")
        print(f"\n[推理] 处理子任�? {subset}")
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if limit:
            data = data[:limit]
            print(f"  [调试] 本子任务处理�?{limit} �?)

        for idx, item in enumerate(tqdm(data, desc=subset)):
            images: List[Image.Image] = []
            total_attempted += 1
            try:
                # 1. 加载图片（兼�?image 为字符串或列表）
                raw_images = item.get("image", [])
                if isinstance(raw_images, str):
                    image_paths = [raw_images]
                elif isinstance(raw_images, list):
                    image_paths = raw_images
                else:
                    image_paths = []

                for img_path in image_paths:
                    resolved = resolve_image_path(img_path, data_dir, json_file)
                    if resolved is not None:
                        images.append(Image.open(resolved).convert("RGB"))

                # ERIQ 官方数据均为视觉输入，若无图直接标记为错误，避免无意义空输出污染结果
                if not images:
                    raise RuntimeError(f"No valid images loaded for sample: {item.get('id', 'unknown')}")

                # 2. 读取问题
                question = item["conversations"][0]["value"]

                # 3. 构建 prompt
                messages = build_messages_for_model(cfg, question, len(images))

                # 4. 模型推理
                infer_start = time.perf_counter()
                base_tokens = int(cfg['inference']['max_new_tokens'])
                token_cap = get_env_int("ERIQ_MAX_NEW_TOKENS_CAP", 0)
                if token_cap > 0:
                    base_tokens = min(base_tokens, token_cap)

                force_max_side = get_env_int("ERIQ_FORCE_MAX_SIDE", 0)
                allow_degraded_retry = os.getenv("ERIQ_ALLOW_DEGRADED_RETRY", "0") == "1"
                attempt_plan = [("base", None, base_tokens)]
                if force_max_side > 0:
                    # 显式指定时才强制缩图
                    attempt_plan = [(f"force-{force_max_side}", force_max_side, base_tokens)]
                elif allow_degraded_retry:
                    # 默认关闭。仅在明确允许时，才启用会影响评分口径的降级重试�?                    attempt_plan.extend([
                        ("downscale-1024", 1024, min(base_tokens, 24)),
                        ("downscale-768", 768, min(base_tokens, 16)),
                        ("downscale-512", 512, min(base_tokens, 8)),
                    ])
                output_text = None
                last_err: Optional[Exception] = None
                for attempt_name, max_side, cur_tokens in attempt_plan:
                    trial_images = images
                    if max_side is not None:
                        trial_images = build_resized_images(images, max_side)
                    try:
                        output_text = infer_with_model(
                            cfg, model, processor, messages, trial_images, cur_tokens
                        )
                        break
                    except Exception as ae:
                        last_err = ae
                        err_msg = str(ae)
                        quant_mode = (cfg.get("loader", {}).get("quantization") or "").lower().strip()
                        # 典型�?bitsandbytes 量化层兼容性问题：'Parameter' object has no attribute 'CB'
                        if (not quant_fallback_done) and quant_mode in {"8bit", "4bit"} and "has no attribute 'CB'" in err_msg:
                            print(
                                f"\n[警告] 检测到 {quant_mode} 量化层兼容性错�? {err_msg}\n"
                                f"      自动回退�?fp16 并重试当前样本�?
                            )
                            cfg_fp16 = copy.deepcopy(cfg)
                            cfg_fp16["loader"].pop("quantization", None)
                            cfg = cfg_fp16
                            model, processor = load_model_and_processor(cfg)
                            quant_fallback_done = True
                            continue
                        if torch.cuda.is_available() and is_cuda_runtime_error(err_msg):
                            print(
                                f"\n[警告] CUDA 异常，执行恢复并进入下一档重�? {attempt_name} -> next"
                            )
                            model, processor = reload_model_after_cuda_error(cfg, model, processor)
                            continue
                        raise
                    finally:
                        if max_side is not None:
                            for img in trial_images:
                                try:
                                    img.close()
                                except Exception:
                                    pass
                if output_text is None:
                    if last_err is not None:
                        raise last_err
                    raise RuntimeError("inference failed without detailed error")
                infer_elapsed = time.perf_counter() - infer_start
                inference_latencies.append(infer_elapsed)

                # 5. 提取答案
                gt = item["conversations"][1]["value"]
                is_yn = gt.strip().lower() in ["yes", "no"]
                pred = extract_answer(output_text, is_yn)

                all_predictions.append({
                    "id": item["id"],
                    "image": item.get("image", []),
                    "conversations": item["conversations"],
                    "prediction": pred,
                    "gt_answer": gt,
                    "raw_output": output_text,
                    "latency_sec": round(infer_elapsed, 4),
                })
                total_success += 1

                # 调试打印�?3 �?                if limit and idx < 3:
                    print(f"  [调试] {item['id']}: raw='{output_text[:50]}...' -> pred='{pred}' gt='{gt}'")

            except Exception as e:
                print(f"\n  [错误] {item.get('id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                total_error += 1

                all_predictions.append({
                    "id": item.get("id", f"{subset}:unknown"),
                    "image": item.get("image", []),
                    "conversations": item.get("conversations", []),
                    "prediction": "",
                    "gt_answer": item["conversations"][1]["value"] if item.get("conversations") else "",
                    "raw_output": f"ERROR: {e}",
                    "latency_sec": None,
                })
            finally:
                # 及时释放 PIL 图片对象，降低主内存占用
                for img in images:
                    try:
                        img.close()
                    except Exception:
                        pass
                images.clear()
                # 可选：样本级显存清理（更稳但更慢），默认关�?                if torch.cuda.is_available() and os.getenv("ERIQ_SAMPLE_EMPTY_CACHE", "0") == "1":
                    safe_cuda_cleanup(sample_level=True)
                if os.getenv("ERIQ_GC_EACH_SAMPLE", "0") == "1":
                    gc.collect()
                cooldown_ms = get_env_int("ERIQ_SAMPLE_COOLDOWN_MS", 0)
                if cooldown_ms > 0:
                    time.sleep(cooldown_ms / 1000.0)
                # �?N 条增量落盘，保障异常中断时可恢复结果
                save_partial_checkpoint()

        # 子任务结束后做一次内存回收，降低后续子任务污�?        gc.collect()
        if torch.cuda.is_available():
            safe_cuda_cleanup(sample_level=False)

    # 保存最终结�?    atomic_write_json(output_path, all_predictions)
    # 最终成功后删除 partial 文件，避免与正式结果混淆
    try:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        pass

    print(f"\n{'='*50}")
    print(f"[完成] 共处�?{total_attempted} 条题目（成功 {total_success} / 失败 {total_error}�?)
    total_wall_time = time.perf_counter() - run_start
    print(f"[完成] 总耗时(含加�?IO): {total_wall_time:.2f} �?)
    if inference_latencies:
        avg_latency = statistics.mean(inference_latencies)
        p50_latency = statistics.median(inference_latencies)
        sorted_lat = sorted(inference_latencies)
        p95_index = min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))
        p95_latency = sorted_lat[p95_index]
        print(f"[完成] 推理耗时统计: 平均 {avg_latency:.3f}s | P50 {p50_latency:.3f}s | P95 {p95_latency:.3f}s")
        print(f"[完成] 估算吞吐: {len(inference_latencies) / max(1e-6, sum(inference_latencies)):.2f} item/s")
    print(f"[完成] 结果保存�? {output_path}")
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[完成] 显存峰�? {peak_mem:.2f} GB")
    print(f"[完成] 评分命令:")
    print(f"       python eval_code/eval_hf.py {output_path}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="ERIQ 配置驱动的通用推理脚本")
    parser.add_argument(
        '--config',
        required=True,
        help='模型配置文件路径 (.yaml)，例�?configs/templates/qwen2.5-vl-7b.yaml'
    )
    parser.add_argument(
        '--data-dir',
        default='/mnt/e/Projects/eriq-eval/data',
        help='ERIQ 数据集目�?(默认: /mnt/e/Projects/eriq-eval/data)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='输出文件路径（默认自动根据配置名命名�?
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='仅测试前 N 条（用于快速调试）'
    )
    parser.add_argument(
        '--subset-file',
        default=None,
        help='仅运行一�?QA 子任务文件，例如 QA_DUALVIEW_MATCHING.json'
    )
    args = parser.parse_args()

    # 加载配置
    config_path = os.path.expanduser(args.config)
    if not os.path.exists(config_path):
        print(f"[错误] 配置文件不存�? {config_path}")
        sys.exit(1)

    cfg = load_config(config_path)
    configure_cuda_attention_backend()

    # 路径处理
    data_dir = os.path.expanduser(args.data_dir)

    if args.output is None:
        model_key = os.path.basename(config_path).replace('.yaml', '')
        args.output = f'/mnt/e/Projects/eriq-eval/results/{model_key}_predictions.json'

    output_path = os.path.expanduser(args.output)

    # 打印启动信息
    print("="*50)
    print(f"模型: {cfg['model_name']}")
    print(f"配置: {config_path}")
    print(f"数据: {data_dir}")
    print(f"输出: {output_path}")
    if args.limit:
        print(f"调试模式: 仅处理前 {args.limit} �?)
    print("="*50)

    # 环境检�?    print(f"\n[环境] PyTorch: {torch.__version__}")
    print(f"[环境] CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[环境] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[环境] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 运行推理
    run_inference(cfg, data_dir, output_path, args.limit, args.subset_file)


if __name__ == '__main__':
    main()
