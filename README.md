# ローカルLLM + RAG ログ要約システム

## プロジェクト構造

```
ollama_setup/
├── src/                    # ソースコード
│   ├── core/              # コアモジュール
│   │   ├── __init__.py
│   │   ├── llm.py         # LLMクラス
│   │   ├── rag.py         # RAGクラス
│   │   └── knowledge_base.py  # ナレッジベースクラス
│   ├── utils/             # ユーティリティ
│   │   ├── __init__.py
│   │   └── log_summarizer.py  # ログ要約システム
│   └── __init__.py
├── config/                # 設定ファイル
│   └── settings.py
├── data/                  # データファイル
│   ├── knowledge_base.csv
│   └── logs/             # ログファイル
├── examples/              # 使用例
│   ├── demo_log_analysis.py
│   └── basic_usage.py
├── requirements.txt       # 依存関係
└── README.md             # このファイル
```

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用例

```python
from src.utils.log_summarizer import LogSummarizer

# ログ要約システムを初期化
summarizer = LogSummarizer()

# ログファイルを読み込み
summarizer.load_log_file("data/logs/app.log")

# エラー分析
error_analysis = summarizer.analyze_errors()
print(error_analysis)
```

### デモの実行

```bash
python examples/demo_log_analysis.py
```

## 機能

- **ローカルLLM**: Hugging Face Transformersを使用
- **RAG (検索拡張生成)**: Faissベクトル検索
- **ナレッジベース**: CSV形式の問題対策データベース
- **ログ要約**: 自動ログ分析と問題対策提案
- **時系列分析**: ログイベントの時間順整理
- **エラー分析**: エラーログの原因と対策分析

## 設定

`config/settings.py`でモデルやパラメータを設定できます。

## ナレッジベース

`data/knowledge_base.csv`に問題対策情報を追加できます。
フォーマット: 問題名,カテゴリ,対処法,詳細説明,予防策,参考情報
