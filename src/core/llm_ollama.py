import requests
from config import settings
import json


class OllamaLLM:
    def __init__(self, model_name=None):
        self.model = model_name or settings.MODEL
        self.default_system_prompt = settings.DEFAULT_SYSTEM_PROMPT
        self.custom_system_prompt = settings.CUSTOM_SYSTEM_PROMPT

    def process(self, messages):
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt},
            proxies={"http": None, "https": None},
            stream=True,  # ストリームで受け取る
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                full_response += data.get("response", "")

        return full_response

    def input_text(self, text):
        messages = [{"role": "system", "content": self.default_system_prompt}]
        if self.custom_system_prompt:
            messages.append({"role": "user", "content": self.custom_system_prompt})
        messages.append({"role": "user", "content": text})
        return self.process(messages)

    def input_text_list(self, texts):
        return [self.input_text(text) for text in texts]

    def summarize_with_context(self, user_request, context_data):
        if isinstance(context_data, list):
            context_text = "\n".join([str(item.get("text", item)) for item in context_data])
        else:
            context_text = str(context_data)

        summary_system_prompt = """あなたは優秀なログ分析・要約の専門家です。
提供されたログデータを分析し、ユーザーの要求に応じて適切な要約を生成してください。
エラーや問題がある場合は重要度を示し、時系列や原因分析も含めてください。
回答は日本語で行ってください。自己紹介は不要です。"""

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
