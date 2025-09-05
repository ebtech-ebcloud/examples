from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import argparse
from transformers import AutoModelForCausalLM, GenerationConfig
from modelscope import AutoTokenizer
import torch


class ChatSession:
    def __init__(self, model_path: str, max_new_tokens=1024):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )

        # 设置生成配置
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.history = []

    def chat(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})

        input_tensor = self.tokenizer.apply_chat_template(
            self.history,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            input_tensor.to(self.model.device),
            max_new_tokens=self.max_new_tokens
        )

        response = self.tokenizer.decode(
            outputs[0][input_tensor.shape[1]:],
            skip_special_tokens=True
        )

        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.history = []


# ---------------- FastAPI 部分 ---------------- #
app = FastAPI()
session = ChatSession("/data/output/deepseek-sft-merged")

class ChatRequest(BaseModel):
    text: str


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    reply = session.chat(request.text)
    return {"reply": reply}


@app.post("/reset")
def reset_endpoint():
    session.reset()
    return {"status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
