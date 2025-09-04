import argparse
from os.path import join
from swanlab.integration.huggingface import SwanLabCallback
import pandas as pd
from datasets import Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import json

# 配置参数
def configuration_parameter():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for deepseek model")

    # 模型路径相关参数
    parser.add_argument("--model_name_or_path", type=str, default="./model",
                        help="Path to the model directory downloaded locally")
    parser.add_argument("--output_dir", type=str,
                        default="/home/public/TrainerShareFolder/lxy/deepseek-output/singledata-2048-16-32-epoch-2",
                        help="Directory to save the fine-tuned model and checkpoints")

    # 数据集路径
    parser.add_argument("--train_file", type=str, default="./data/single_datas.jsonl",
                        help="Path to the training data file in JSONL format")

    # 训练超参数
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the input")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of steps between logging metrics")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                        help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduler")

    # LoRA 特定参数
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")

    # 额外优化和硬件相关参数
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Use mixed precision (FP16) training")
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Strategy for saving checkpoints ('steps', 'epoch')")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay for the optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1,
                        help="Maximum gradient norm for clipping")

    args = parser.parse_args()
    return args

def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            lora_module_names.add(name.split('.')[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names

# 加载模型
def load_model(args, train_dataset, data_collator):
    # 自动分配设备
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
        "use_cache": False if args.gradient_checkpointing else True,
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    # 用于确保模型的词嵌入层参与训练
    model.enable_input_require_grads()

    # 哪些模块需要注入Lora参数
    target_modules = find_all_linear_names(model.module if isinstance(model, DDP) else model, args.train_mode)
    # lora参数设置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False

    )
    use_bfloat16 = torch.cuda.is_bf16_supported()  # 检查设备是否支持 bf16
    # 配置训练参数
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        optim=args.optim,
        ddp_find_unused_parameters=False,  # 分布式参数检查优化
        fp16=args.fp16,
        bf16=not args.fp16 and use_bfloat16,
        remove_unused_columns=False
    )
    # 应用 PEFT 配置到模型
    model = get_peft_model(model.module if isinstance(model, DDP) else model, config)  # 确保传递的是原始模型
    model.print_trainable_parameters()

    ### 展示平台
    swanlab_config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dataset": "single-data-3w"

    }
    callbacks = []
    if SwanLabCallback is not None:
        callbacks.append(SwanLabCallback(
          project="deepseek-finetune",
          experiment_name="deepseek-llm-7b-chat-lora",
          description="DeepSeek有很多模型，V2太大了，这里选择llm-7b-chat的，希望能让回答更加人性化",
          workspace=None,
          config=swanlab_config,
    ))
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    return trainer

# 处理数据
def process_data(data: dict, tokenizer, max_seq_length):
    """
    把 conversation 里的 human/assistant 拼接成训练样本
    格式: "Human: ... \n\nAssistant: ...<eos>"
    """
    input_ids, attention_mask, labels = [], [], []

    for conv in data["conversation"]:
        text = f"Human: {conv['human'].strip()}\n\nAssistant: {conv['assistant'].strip()}{tokenizer.eos_token}"
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        input_ids += tokenized["input_ids"]
        attention_mask += tokenized["attention_mask"]
        labels += tokenized["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 训练部分
def main():
    args = configuration_parameter()
    print("*****************加载分词器*************************")
    # 加载分词器
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print("*****************处理数据*************************")
    # 处理数据
    # 获得数据
    data = pd.read_json(args.train_file, lines=True)
    train_ds = Dataset.from_pandas(data)
    train_dataset = train_ds.map(process_data,
                                 fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
                                 remove_columns=train_ds.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    print(train_dataset, data_collator)
    # 加载模型
    print("*****************train*************************")
    trainer = load_model(args, train_dataset, data_collator)
    trainer.train()
    # 训练
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)

if __name__ == "__main__":
    main()
