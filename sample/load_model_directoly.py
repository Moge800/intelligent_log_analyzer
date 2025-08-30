# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os

# settingsをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import settings

# GPU設定の取得
use_gpu = settings.USE_GPU and torch.cuda.is_available()

# デバイス設定
if use_gpu:
    if settings.GPU_DEVICE == "auto":
        device_map = "auto"
    else:
        device_map = settings.GPU_DEVICE
else:
    device_map = None

# データ型設定
if settings.TORCH_DTYPE == "auto":
    torch_dtype = torch.float16 if use_gpu else torch.float32
elif settings.TORCH_DTYPE == "float16":
    torch_dtype = torch.float16
elif settings.TORCH_DTYPE == "float32":
    torch_dtype = torch.float32
elif settings.TORCH_DTYPE == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(settings.MODEL)
model = AutoModelForCausalLM.from_pretrained(
    settings.MODEL,
    torch_dtype=torch_dtype,
    device_map=device_map,
)
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=settings.DEFAULT_MAX_TOKENS)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
