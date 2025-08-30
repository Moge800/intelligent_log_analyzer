"""
基本的な使用例
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.log_summarizer import LogSummarizer


def main():
    """基本的な使用例"""
    print("=== 基本的なログ要約システムの使用例 ===\n")

    # ログ要約システムを初期化
    summarizer = LogSummarizer()

    # サンプルログを作成
    sample_log = """2024-08-30 14:00:01 INFO システム起動
2024-08-30 14:00:05 ERROR データベース接続失敗: timeout
2024-08-30 14:00:10 INFO 再試行中...
2024-08-30 14:00:15 INFO データベース接続成功"""

    # サンプルログファイルを作成
    os.makedirs("./temp_logs", exist_ok=True)
    with open("./temp_logs/sample.log", "w", encoding="utf-8") as f:
        f.write(sample_log)

    # ログファイルを読み込み
    print("1. ログファイルを読み込み中...")
    summarizer.load_log_file("./temp_logs/sample.log")

    # エラー分析
    print("\n2. エラー分析:")
    print("-" * 40)
    error_analysis = summarizer.analyze_errors()
    print(error_analysis)

    # 時系列分析
    print("\n3. 時系列分析:")
    print("-" * 40)
    timeline = summarizer.get_timeline_summary()
    print(timeline)

    # ナレッジベース検索
    print("\n4. 関連する解決策:")
    print("-" * 40)
    solutions = summarizer.search_problem_solutions(["データベース", "接続"])
    print(solutions)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
