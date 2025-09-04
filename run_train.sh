#!/bin/bash

python train.py \
  --model_name_or_path /public/huggingface-models/deepseek-ai/deepseek-llm-7b-chat \
  --train_file /public/github/EmoLLM/datasets/mother_v1.json \
  --output_dir /data/output/deepseek-7b-lora \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --train_mode lora \
  --fp16 True
