"""
ログ要約システムの使用例

このスクリプトでは、以下のことができます：
1. ログファイルをRAGシステムに読み込み
2. 特定の要求に基づいてログを要約
3. エラー分析、パフォーマンス分析
4. 時系列での要約
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.log_summarizer import LogSummarizer
import os


def create_sample_logs():
    """サンプルログファイルを作成"""
    os.makedirs("./logs", exist_ok=True)

    # Webサーバーログのサンプル
    web_log = """2024-08-30 09:00:01 INFO [WebServer] サーバー起動開始
2024-08-30 09:00:02 INFO [WebServer] ポート8080でリスニング開始
2024-08-30 09:15:30 INFO [Request] GET /api/users - 200 OK (45ms)
2024-08-30 09:16:15 WARNING [Request] GET /api/data - 응답시간이 500ms를 초과했습니다 (750ms)
2024-08-30 09:17:45 ERROR [Database] Connection pool exhausted - 最大接続数に達しました
2024-08-30 09:17:46 INFO [Database] 接続プールを拡張しています...
2024-08-30 09:17:50 INFO [Database] 接続プール拡張完了 (20 -> 30 connections)
2024-08-30 09:20:30 INFO [Request] POST /api/login - 200 OK (120ms)
2024-08-30 09:25:10 ERROR [Authentication] ログイン失敗: 無効なパスワード (user: admin)
2024-08-30 09:25:15 WARNING [Security] 3回連続ログイン失敗 - IPアドレス: 192.168.1.100
2024-08-30 09:30:00 INFO [Cleanup] 古いセッションを削除しました (削除数: 25)"""

    # アプリケーションログのサンプル
    app_log = """2024-08-30 10:00:01 INFO [App] アプリケーション起動
2024-08-30 10:00:05 INFO [Config] 設定ファイル読み込み完了
2024-08-30 10:00:10 INFO [Database] データベース接続確立
2024-08-30 10:01:00 INFO [Processing] バッチ処理開始 - 対象件数: 10000件
2024-08-30 10:01:30 WARNING [Memory] メモリ使用量: 85% (1.7GB/2GB)
2024-08-30 10:02:15 ERROR [Processing] レコード処理エラー: ID=5432, Reason=Invalid data format
2024-08-30 10:02:16 INFO [Processing] エラーレコードをスキップして処理継続
2024-08-30 10:05:45 INFO [Processing] バッチ処理完了 - 成功: 9998件, エラー: 2件
2024-08-30 10:06:00 INFO [Maintenance] 自動バックアップ開始
2024-08-30 10:08:30 INFO [Maintenance] バックアップ完了 - サイズ: 2.3GB"""

    with open("./logs/web_server.log", "w", encoding="utf-8") as f:
        f.write(web_log)

    with open("./logs/application.log", "w", encoding="utf-8") as f:
        f.write(app_log)

    print("サンプルログファイルを作成しました:")
    print("- ./logs/web_server.log")
    print("- ./logs/application.log")


def main():
    print("=== ログ要約システム デモ ===\n")

    # サンプルログファイルを作成
    create_sample_logs()

    # ログ要約システムを初期化
    print("ログ要約システムを初期化中...")
    summarizer = LogSummarizer()

    # ログディレクトリを読み込み
    print("\nログファイルを読み込み中...")
    summarizer.load_log_directory("./logs")

    print("\n" + "=" * 50)
    print("各種分析を実行します...")
    print("=" * 50)

    # 1. エラー分析
    print("\n【1. エラー分析】")
    print("-" * 30)
    error_analysis = summarizer.analyze_errors()
    print(error_analysis)

    # 2. パフォーマンス分析
    print("\n【2. パフォーマンス分析】")
    print("-" * 30)
    performance_analysis = summarizer.analyze_performance()
    print(performance_analysis)

    # 3. 時系列要約
    print("\n【3. 時系列要約】")
    print("-" * 30)
    timeline_summary = summarizer.get_timeline_summary()
    print(timeline_summary)

    # 4. カスタム要約例
    print("\n【4. データベース関連問題の分析】")
    print("-" * 30)
    db_analysis = summarizer.summarize_logs("データベース関連の問題やパフォーマンスについて詳しく分析してください")
    print(db_analysis)

    # 5. セキュリティ関連分析
    print("\n【5. セキュリティ関連分析】")
    print("-" * 30)
    security_analysis = summarizer.summarize_logs("セキュリティやログイン失敗について分析してください")
    print(security_analysis)

    # 6. ナレッジベース直接検索
    print("\n【6. ナレッジベース検索（データベース問題）】")
    print("-" * 30)
    kb_solutions = summarizer.search_problem_solutions(["データベース", "接続"])
    print(kb_solutions)

    # 7. 新しい解決策の追加（例）
    print("\n【7. 新しい解決策の追加】")
    print("-" * 30)
    add_result = summarizer.add_new_solution(
        problem_name="Webサーバー応答遅延",
        category="パフォーマンス",
        solution="負荷分散とキャッシュ設定の最適化",
        details="同時接続数の増加によるボトルネック",
        prevention="監視アラートと自動スケーリング設定",
        reference="負荷テスト結果とパフォーマンス分析",
    )
    print(add_result)

    # 8. データを保存
    print("\n【8. RAGデータを保存】")
    print("-" * 30)
    save_success = summarizer.rag.save_all("./log_analysis_results")
    if save_success:
        print("RAGデータの保存が完了しました。")
    else:
        print("RAGデータの保存に失敗しました。")

    # 9. ナレッジベースを保存
    print("\n【9. ナレッジベースを保存】")
    print("-" * 30)
    kb_save_success = summarizer.save_knowledge_base("./updated_knowledge_base.csv")
    if kb_save_success:
        print("ナレッジベースの保存が完了しました。")
    else:
        print("ナレッジベースの保存に失敗しました。")

    print("\n" + "=" * 50)
    print("分析完了！")
    print("=" * 50)


if __name__ == "__main__":
    main()
