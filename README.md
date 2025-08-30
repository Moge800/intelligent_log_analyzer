# ローカルLLM + RAG ログ要約システム

日本語LLMとRAG（検索拡張生成）を使用したインテリジェントなログ分析・要約システムです。
ローカル環境で動作し、ログファイルから重要な情報を抽出し、問題の診断と対策を自動で提案します。

## プロジェクト構造

```
intelligent_log_analyzer/
├── src/                    # ソースコード
│   ├── core/              # コアモジュール
│   │   ├── __init__.py
│   │   ├── llm.py         # LLMクラス (Hugging Face Transformers)
│   │   ├── rag.py         # RAGクラス (Faiss vectorベース)
│   │   └── knowledge_base.py  # ナレッジベースクラス
│   ├── utils/             # ユーティリティ
│   │   ├── __init__.py
│   │   ├── log_summarizer.py  # ログ要約システム
│   │   └── gpu_test.py    # GPU環境テストツール
│   └── __init__.py
├── config/                # 設定ファイル
│   └── settings.py        # モデル・GPU設定
├── data/                  # データファイル
│   └── knowledge_base.csv # 問題対策データベース
├── examples/              # 使用例とデモ
│   ├── demo_log_analysis.py  # デモスクリプト
│   └── basic_usage.py     # 基本使用例
├── sample/                # サンプルコード
│   ├── jp_model_test.py
│   ├── pipe_test.py
│   └── load_model_directoly.py
├── main.py               # メインエントリーポイント
├── requirements.txt      # 依存関係
├── LICENSE
└── README.md            # このファイル
```

## システム要件

- Python 3.8以上
- GPU（推奨）: CUDA対応GPU、8GB以上のVRAM
- メモリ: 16GB以上推奨
- ディスク容量: モデルファイル用に20GB以上の空き容量

## インストール

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. GPU環境の確認（推奨）

```bash
# GPU環境テスト（適切なPYTHONPATH設定付き）
PYTHONPATH=/path/to/intelligent_log_analyzer python src/utils/gpu_test.py
```

### 3. 初回実行時の注意

初回実行時は、Hugging FaceからELYZA-JP-8Bモデル（約16GB）がダウンロードされます。
インターネット接続が必要です。

```bash
# メイン実行（適切なPYTHONPATH設定付き）
PYTHONPATH=/path/to/intelligent_log_analyzer python main.py
```

## 使用方法

### メインアプリケーションの実行

```bash
# デモログを生成して分析実行（適切なPYTHONPATH設定付き）
PYTHONPATH=/path/to/intelligent_log_analyzer python main.py
```

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

# 時系列分析
timeline = summarizer.get_timeline_summary()
print(timeline)

# カスタム要約
custom_summary = summarizer.summarize_logs("データベースエラーに関する情報を要約してください")
print(custom_summary)
```

### デモの実行

```bash
# 基本使用例（適切なPYTHONPATH設定付き）
PYTHONPATH=/path/to/intelligent_log_analyzer python examples/basic_usage.py

# 詳細なデモ（適切なPYTHONPATH設定付き）
PYTHONPATH=/path/to/intelligent_log_analyzer python examples/demo_log_analysis.py
```

## 機能

### 主要機能
- **ローカルLLM**: ELYZA-JP-8B日本語大規模言語モデル
- **RAG (検索拡張生成)**: Faissベクトル検索による関連ログ抽出
- **ナレッジベース**: CSV形式の問題対策データベース統合
- **ログ要約**: ユーザー要求に応じた自動ログ分析
- **時系列分析**: ログイベントの時間順整理と流れの可視化
- **エラー分析**: エラーログの原因分析と対策提案
- **問題解決提案**: ナレッジベースに基づく具体的対策

### 技術スタック
- **LLM**: Hugging Face Transformers + ELYZA-JP-8B
- **ベクトル検索**: Faiss (Facebook AI Similarity Search)
- **GPU対応**: CUDA/PyTorch + 量子化サポート
- **言語**: Python 3.8+

## 設定

`config/settings.py`でモデルやパラメータを詳細に設定できます。

### モデル設定
```python
MODEL = "elyza/Llama-3-ELYZA-JP-8B"  # 使用するHugging Faceモデル
DEFAULT_MAX_TOKENS = 1200             # 生成する最大トークン数
DEFAULT_SYSTEM_PROMPT = "..."         # システムプロンプト
```

### GPU設定
```python
USE_GPU = True          # GPUを使用するかどうか
FORCE_GPU = False       # GPUが利用できない場合でもエラーを出すかどうか
GPU_DEVICE = "auto"     # "auto", "cuda:0", "cuda:1", "balanced", "balanced_low_0" など
TORCH_DTYPE = "auto"    # "auto", "float16", "float32", "bfloat16" など
QUANTIZATION = None     # None, "8bit", "4bit" (メモリ節約)
```

### 推奨設定
- **高性能GPU（24GB+ VRAM）**: `TORCH_DTYPE = "float16"`, `QUANTIZATION = None`
- **中性能GPU（8-16GB VRAM）**: `TORCH_DTYPE = "float16"`, `QUANTIZATION = "8bit"`
- **低性能GPU（4-8GB VRAM）**: `TORCH_DTYPE = "float16"`, `QUANTIZATION = "4bit"`
- **CPU実行**: `USE_GPU = False`

## ナレッジベース

`data/knowledge_base.csv`に問題対策情報を追加・管理できます。

### ファイル形式
```csv
問題名,カテゴリ,対処法,詳細説明,予防策,参考情報
データベース接続エラー,データベース,接続プールの設定を確認し再起動を試行,接続タイムアウトやプール枯渇が原因の可能性。max_connections設定を確認,定期的な接続プール監視とアラート設定,DB接続数の監視ダッシュボード導入
```

### プログラムからの操作
```python
from src.core.knowledge_base import KnowledgeBase

kb = KnowledgeBase("data/knowledge_base.csv")

# 新しい問題対策を追加
kb.add_knowledge_entry(
    problem_name="API応答遅延",
    category="API", 
    solution="レート制限とキャッシュ設定を確認",
    details="外部API呼び出しでのタイムアウト",
    prevention="適切なレート制限とキャッシュ戦略",
    reference="API監視ツールの導入"
)

# 問題検索
solutions = kb.search_solutions(["データベース", "接続"])
```

## トラブルシューティング

### よくある問題

#### 1. GPU関連エラー
```bash
# GPU環境確認（適切なPYTHONPATH設定付き）
PYTHONPATH=/path/to/intelligent_log_analyzer python src/utils/gpu_test.py

# GPU使用しない場合
# config/settings.py で USE_GPU = False に設定
```

#### 2. メモリ不足エラー
```python
# config/settings.py で量子化を有効化
QUANTIZATION = "8bit"  # または "4bit"
TORCH_DTYPE = "float16"
```

#### 3. モデルダウンロードエラー
- インターネット接続を確認
- Hugging Face Hubへのアクセスを確認
- プロキシ環境の場合、適切な設定を行う

#### 4. インポートエラー
```bash
# PYTHONPATH を適切に設定
export PYTHONPATH=/path/to/intelligent_log_analyzer:$PYTHONPATH
```

### パフォーマンス最適化

1. **GPU設定の最適化**: 使用可能なVRAMに応じて量子化レベルを調整
2. **バッチサイズ調整**: メモリ使用量に応じてRAGの検索件数を調整
3. **モデル選択**: 用途に応じてより軽量なモデルに変更可能

## ライセンス

このプロジェクトはLICENSEファイルに従います。

## 貢献

プルリクエストやイシューの報告を歓迎します。バグ報告や機能提案がある場合は、GitHubのIssuesをご利用ください。
