import argparse
import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from modelscope import AutoTokenizer

EXCLUDE_DIRS = {"__pycache__", ".git"}

def copy_files_not_in_B(A_path, B_path):
    """复制 A 中除权重外的文件/目录到 B"""
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    files_in_A = [
        f for f in os.listdir(A_path)
        if not (".bin" in f or "safetensors" in f or f in EXCLUDE_DIRS)
    ]
    files_in_B = set(os.listdir(B_path))
    files_to_copy = set(files_in_A) - files_in_B

    for file in files_to_copy:
        src, dst = os.path.join(A_path, file), os.path.join(B_path, file)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

def merge_lora_to_base_model(base_model, adapter_model, save_path):
    # 处理保存目录
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_model, device_map="auto", trust_remote_code=True)

    # 合并 LoRA 权重
    merged_model = model.merge_and_unload()

    # 保存
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path, safe_serialization=False)
    copy_files_not_in_B(base_model, save_path)
    print(f"合并后的模型已保存至: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="原始模型路径")
    parser.add_argument("--adapter_model", type=str, required=True, help="LoRA 微调模型路径")
    parser.add_argument("--save_path", type=str, required=True, help="合并后模型保存路径")
    args = parser.parse_args()

    merge_lora_to_base_model(args.base_model, args.adapter_model, args.save_path)
