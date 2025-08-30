import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# settingsをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import settings

text = "elyza/Llama-3-ELYZA-JP-8Bのファイル詳細を教えて下さい"

model_name = settings.MODEL

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

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map=device_map,
)
model.eval()

messages = [
    {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": text},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=settings.DEFAULT_MAX_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
print(output)
