MODEL = "elyza/Llama-3-ELYZA-JP-8B"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_SYSTEM_PROMPT = (
    "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
)
CUSTOM_SYSTEM_PROMPT = ""  # Add your custom system prompt here if needed

# GPU設定
USE_GPU = True  # GPUを使用するかどうか
FORCE_GPU = False  # GPUが利用できない場合でもエラーを出すかどうか
GPU_DEVICE = "auto"  # "auto", "cuda:0", "cuda:1", "balanced", "balanced_low_0" など
TORCH_DTYPE = "auto"  # "auto", "float16", "float32", "bfloat16" など
QUANTIZATION = None  # None, "8bit", "4bit"
LOW_MEMORY = False  # Low memory mode
