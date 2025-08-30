# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
import sys
import os

# settingsをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import settings

# パイプライン用のデバイス設定
device = 0 if settings.USE_GPU and torch.cuda.is_available() else -1

# データ型設定
if settings.TORCH_DTYPE == "auto":
    torch_dtype = torch.float16 if device >= 0 else torch.float32
elif settings.TORCH_DTYPE == "float16":
    torch_dtype = torch.float16
elif settings.TORCH_DTYPE == "float32":
    torch_dtype = torch.float32
elif settings.TORCH_DTYPE == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

pipe = pipeline("text-generation", model=settings.MODEL, device=device, torch_dtype=torch_dtype)
messages = [
    {"role": "user", "content": "Who are you?"},
]
response = pipe(messages, max_new_tokens=settings.DEFAULT_MAX_TOKENS)

print(response)
