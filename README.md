# Quick Start

## 下载代码
```
git clone https://github.com/ebtech-ebcloud/quick_start.git
cd quick_start
```

## 安装依赖

在项目根目录下执行：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
如果版本过低，升级 PyTorch
```
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## 训练模型
以 **Deepseek-LLM** 微调示例
请确保此次微调 **显存至少有 15GB**，可使用 **4090** 进行测试

模型及数据集地址

`/public/huggingface-models/deepseek-ai/deepseek-llm-7b-chat/`

`/public/github/EmoLLM/`

运行 `run_train.sh` 进行训练
等待训练结束，训练过程演示可以在Swanlab上看，会输出如下图片


训练结束之后，需要把保存下来的模型与预训练模型做一个合并
```
python modeladd.py \
  --base_model /data/deepseek-llm-7b-chat \
  --adapter_model /data/output/deepseek-7b-lora \
  --save_path /data/output/deepseek-sft-merged
```

