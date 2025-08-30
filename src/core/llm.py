import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from ..core.rag import RAG

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import settings
import warnings
import logging

# 警告メッセージを抑制
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLM:
    def __init__(self, model_name):
        # settings.pyからGPU設定を取得
        use_gpu = settings.USE_GPU and torch.cuda.is_available()

        # GPUが強制されているが利用できない場合はエラー
        if settings.FORCE_GPU and not torch.cuda.is_available():
            raise RuntimeError("GPU is forced but CUDA is not available")

        # デバイス設定
        if use_gpu:
            device = "cuda"
            # GPU_DEVICE設定を使用
            if settings.GPU_DEVICE == "auto":
                device_map = "auto"
            else:
                device_map = settings.GPU_DEVICE
        else:
            device = "cpu"
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
            torch_dtype = torch.float32  # デフォルト

        # 量子化設定
        load_in_8bit = settings.QUANTIZATION == "8bit"
        load_in_4bit = settings.QUANTIZATION == "4bit"

        # モデルロード設定を構築
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # 手動でデバイスに移動する場合（device_mapを使わない場合）
        if device_map is None and use_gpu:
            self.model = self.model.to(device)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPU使用状況をログ出力
        print(f"🔧 GPU設定:")
        print(f"   CUDA利用可能: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数: {torch.cuda.device_count()}")
            print(f"   現在のGPU: {torch.cuda.current_device()}")
            print(f"   GPU名: {torch.cuda.get_device_name()}")
        print(f"   USE_GPU設定: {settings.USE_GPU}")
        print(f"   実際にGPU使用: {use_gpu}")
        print(f"   デバイスマップ: {device_map}")
        print(f"   データ型: {torch_dtype}")
        print(f"   モデルのデバイス: {next(self.model.parameters()).device}")

        # pad_tokenを設定して警告を抑制
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.default_system_prompt = settings.DEFAULT_SYSTEM_PROMPT
        self.custom_system_prompt = settings.CUSTOM_SYSTEM_PROMPT
        self.max_tokens = settings.DEFAULT_MAX_TOKENS

    def process(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(token_ids),
            )
        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        return output

    def print_info(self):
        print(f"Model: {self.model}")
        # print(f"Tokenizer: {self.tokenizer}")

    def input_text(self, text):
        messages = [{"role": "system", "content": self.default_system_prompt}]
        if self.custom_system_prompt:
            messages.append({"role": "user", "content": self.custom_system_prompt})
        messages.append({"role": "user", "content": text})
        return self.process(messages)

    def input_text_list(self, texts) -> list:
        return [self.input_text(text) for text in texts]

    def summarize_with_context(self, user_request, context_data):
        """文脈データを使用して要約生成"""
        # 文脈データを文字列に変換
        if isinstance(context_data, list):
            context_text = "\n".join([str(item.get("text", item)) for item in context_data])
        else:
            context_text = str(context_data)

        # システムプロンプトを要約用に調整
        summary_system_prompt = """あなたは優秀なログ分析・要約の専門家です。
提供されたログデータを分析し、ユーザーの要求に応じて適切な要約を生成してください。
エラーや問題がある場合は重要度を示し、時系列や原因分析も含めてください。
回答は日本語で行ってください。"""

        messages = [
            {"role": "system", "content": summary_system_prompt},
            {
                "role": "user",
                "content": f"以下のデータを分析して要約してください。\n\nユーザーの要求: {user_request}\n\nデータ:\n{context_text}",
            },
        ]

        return self.process(messages)

    def input_text_and_vector(self, text, vector):
        messages = [{"role": "system", "content": self.default_system_prompt}]
        if self.custom_system_prompt:
            messages.append({"role": "user", "content": self.custom_system_prompt})
        messages.append({"role": "user", "content": text, "vector": vector})
        return self.process(messages)


if __name__ == "__main__":
    llm = LLM(model_name=settings.MODEL)
    # llm.print_info()
    print(llm.input_text("Hello, world!"))
    print(llm.input_text_list(["Hello, world!", "How are you?"]))
    print(llm.input_text("ぬるぽ"))
    print("Done")

    # RAGのテスト
    rag = RAG()
    # データを追加してからクエリを実行
    rag.add_text("Hello world greeting message")
    rag.add_text("Python programming language")
    rag.add_text("Machine learning and AI")

    query_results = rag.query("Hello, world!")
    print(f"RAG query results: {query_results}")
    print(llm.input_text_and_vector("Hello, world!", query_results))
