import os
import re
from pathlib import Path
from ..core.rag import RAG
from ..core.knowledge_base import KnowledgeBase
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import settings


class LogSummarizer:
    def __init__(self, model_name=None, knowledge_base_path="data/knowledge_base.csv"):
        """ログ要約システムの初期化"""
        if settings.LLM_BACKEND == "transformers":
            from ..core.llm import LLM

            self.llm = LLM(model_name or settings.MODEL)
        else:
            from ..core.llm_ollama import OllamaLLM

            self.llm = OllamaLLM(model_name or settings.MODEL)

        self.rag = RAG()
        self.knowledge_base = KnowledgeBase(knowledge_base_path)
        self.log_patterns = {
            "error": r"(?i)(error|エラー|exception|失敗|異常)",
            "warning": r"(?i)(warning|warn|警告|注意|警報)",
            "info": r"(?i)(info|information|情報|確認)",
            "timestamp": r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}",
        }

    def load_log_file(self, file_path):
        """単一ログファイルを読み込んでRAGに追加"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # ログをチャンクに分割（行ごと、または時間戳で区切る）
            chunks = self._split_log_content(content)

            for chunk in chunks:
                if chunk.strip():  # 空でないチャンクのみ追加
                    self.rag.add_text(chunk)

            print(f"ログファイル '{file_path}' を読み込みました。{len(chunks)}個のチャンクを追加。")
            return True

        except Exception as e:
            print(f"ファイル読み込みエラー: {e}")
            return False

    def load_log_directory(self, directory_path, file_pattern="*.log"):
        """ディレクトリ内のログファイルを一括読み込み"""
        loaded_count = 0
        for file_path in Path(directory_path).rglob(file_pattern):
            if self.load_log_file(file_path):
                loaded_count += 1

        print(f"合計 {loaded_count} 個のログファイルを読み込みました。")
        return loaded_count > 0

    def _split_log_content(self, content):
        """ログ内容を適切なチャンクに分割"""
        # タイムスタンプで分割を試行
        timestamp_pattern = self.log_patterns["timestamp"]
        lines = content.split("\n")
        chunks = []
        current_chunk = []

        for line in lines:
            if re.search(timestamp_pattern, line) and current_chunk:
                # 新しいタイムスタンプが見つかったら、前のチャンクを完成
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # 最後のチャンクを追加
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # チャンクが長すぎる場合は行数で分割
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 1000:  # 1000文字を超える場合
                sub_chunks = [chunk[i : i + 800] for i in range(0, len(chunk), 800)]
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def summarize_logs(self, user_request="ログの内容を要約してください"):
        """ユーザーの要求に基づいてログを要約（ナレッジベース統合）"""
        # RAGで関連するログエントリを検索
        relevant_logs = self.rag.query(user_request, k=10)

        if not relevant_logs:
            return "関連するログエントリが見つかりませんでした。"

        # 関連ログを文脈として組み合わせ
        context = self._build_context(relevant_logs)

        # ログから問題キーワードを抽出してナレッジベース検索
        knowledge_solutions = self._search_knowledge_for_logs(relevant_logs, user_request)

        # ナレッジベースの情報を含むプロンプトを構築
        summary_prompt = self._build_enhanced_summary_prompt(user_request, context, knowledge_solutions)

        # LLMで要約生成
        summary = self.llm.input_text(summary_prompt)

        return summary

    def _search_knowledge_for_logs(self, relevant_logs, user_request):
        """ログエントリからナレッジベースで関連する解決策を検索"""
        all_solutions = []

        # ログエントリからキーワードを抽出
        keywords = set()
        for log_entry in relevant_logs:
            log_text = log_entry["text"].lower()

            # 一般的な問題キーワードを抽出
            problem_keywords = [
                "データベース",
                "database",
                "メモリ",
                "memory",
                "ネットワーク",
                "network",
                "エラー",
                "error",
                "接続",
                "connection",
                "タイムアウト",
                "timeout",
                "ログイン",
                "login",
                "ssl",
                "api",
                "バックアップ",
                "backup",
                "ディスク",
                "disk",
                "プロセス",
                "process",
            ]

            for keyword in problem_keywords:
                if keyword in log_text:
                    keywords.add(keyword)

        # ユーザーリクエストからもキーワード抽出
        request_text = user_request.lower()
        for keyword in ["エラー", "error", "パフォーマンス", "performance", "セキュリティ", "security"]:
            if keyword in request_text:
                keywords.add(keyword)

        # ナレッジベースで検索
        if keywords:
            solutions = self.knowledge_base.search_solutions(list(keywords))
            all_solutions.extend(solutions[:3])  # 上位3件を使用

        return all_solutions

    def _build_enhanced_summary_prompt(self, user_request, context, knowledge_solutions):
        """ナレッジベースの情報を含む拡張プロンプトを構築"""
        knowledge_text = ""
        if knowledge_solutions:
            knowledge_text = "\n\n【関連する対策情報】:\n"
            for i, solution in enumerate(knowledge_solutions, 1):
                knowledge_text += f"{i}. {self.knowledge_base.format_solution(solution)}\n"

        return f"""以下のログエントリとナレッジベースに基づいて、ユーザーの要求に答えてください。

ユーザーの要求: {user_request}

関連するログエントリ:
{context}{knowledge_text}

上記の情報を分析して、ユーザーの要求に適した要約を提供してください。
エラーや問題がある場合は重要度を示し、時系列や原因分析、具体的な対策を含めてください。
ナレッジベースの対策情報がある場合は、それを参考にして実用的な解決案を提示してください。"""

    def _build_context(self, relevant_logs):
        """関連ログから文脈を構築"""
        context_parts = []
        for i, log_entry in enumerate(relevant_logs[:5]):  # 上位5件を使用
            context_parts.append(f"ログエントリ {i+1} (類似度: {log_entry['distance']:.2f}):\n{log_entry['text']}")

        return "\n\n".join(context_parts)

    def _build_summary_prompt(self, user_request, context):
        """要約用のプロンプトを構築"""
        return f"""以下のログエントリに基づいて、ユーザーの要求に答えてください。

ユーザーの要求: {user_request}

関連するログエントリ:
{context}

上記のログエントリを分析して、ユーザーの要求に適した要約を提供してください。
エラーや問題がある場合は重要度を示し、時系列や原因分析も含めてください。"""

    def analyze_errors(self):
        """エラーログの分析"""
        return self.summarize_logs("エラーや例外について分析してください。原因と対策を含めて要約してください。")

    def analyze_performance(self):
        """パフォーマンス関連の分析"""
        return self.summarize_logs("パフォーマンスや処理時間に関する問題を分析してください。")

    def get_timeline_summary(self):
        """時系列での要約"""
        return self.summarize_logs("時系列順でログの流れを要約してください。重要なイベントを時間順に整理してください。")

    def search_problem_solutions(self, problem_keywords):
        """特定の問題に対する解決策を直接検索"""
        solutions = self.knowledge_base.search_solutions(problem_keywords)
        if not solutions:
            return "該当する解決策が見つかりませんでした。"

        result = "【検索された解決策】:\n\n"
        for i, solution in enumerate(solutions[:3], 1):
            result += f"{i}. {self.knowledge_base.format_solution(solution)}\n\n"

        return result

    def add_new_solution(self, problem_name, category, solution, details="", prevention="", reference=""):
        """新しい対策をナレッジベースに追加"""
        success = self.knowledge_base.add_knowledge_entry(
            problem_name, category, solution, details, prevention, reference
        )
        if success:
            return f"新しい解決策を追加しました: {problem_name}"
        else:
            return "解決策の追加に失敗しました。"

    def save_knowledge_base(self, output_path=None):
        """ナレッジベースを保存"""
        return self.knowledge_base.save_to_csv(output_path)


def main():
    """メイン実行関数"""
    print("=== ログ要約システム ===")

    # ログ要約システムを初期化
    summarizer = LogSummarizer()

    # サンプルログファイルの作成（テスト用）
    sample_log_content = """2024-08-30 10:00:01 INFO アプリケーション開始
2024-08-30 10:00:05 INFO データベース接続確立
2024-08-30 10:01:15 WARNING メモリ使用量が80%を超えました
2024-08-30 10:02:30 ERROR データベース接続エラー: Connection timeout
2024-08-30 10:02:31 INFO 再接続を試行中...
2024-08-30 10:02:35 INFO データベース再接続成功
2024-08-30 10:05:00 INFO 処理完了: 1000件のレコードを処理
2024-08-30 10:05:01 WARNING 一時的なネットワーク遅延を検出
2024-08-30 10:06:00 INFO アプリケーション正常終了"""

    # サンプルログファイルを作成
    os.makedirs("./sample_logs", exist_ok=True)
    with open("./sample_logs/app.log", "w", encoding="utf-8") as f:
        f.write(sample_log_content)

    # ログファイルを読み込み
    print("\n1. ログファイルを読み込み中...")
    summarizer.load_log_file("./sample_logs/app.log")

    # 様々な要約を実行
    print("\n2. エラー分析:")
    error_summary = summarizer.analyze_errors()
    print(error_summary)

    print("\n3. 時系列要約:")
    timeline_summary = summarizer.get_timeline_summary()
    print(timeline_summary)

    print("\n4. カスタム要約:")
    custom_summary = summarizer.summarize_logs("データベース関連の問題について詳しく教えてください")
    print(custom_summary)


if __name__ == "__main__":
    main()
