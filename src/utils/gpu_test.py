#!/usr/bin/env python3
"""
GPU使用状況テストスクリプト
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

print("🔍 GPU環境チェック開始...")

try:
    import torch

    print(f"✅ PyTorch インポート成功")
    print(f"   - PyTorchバージョン: {torch.__version__}")
    print(f"   - CUDA利用可能: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   - GPU数: {torch.cuda.device_count()}")
        print(f"   - 現在のGPU: {torch.cuda.current_device()}")
        print(f"   - GPU名: {torch.cuda.get_device_name()}")
        print(f"   - CUDAバージョン: {torch.version.cuda}")

        # 簡単なテンソル作成テスト
        test_tensor = torch.randn(3, 3)
        print(f"   - CPUテンソルデバイス: {test_tensor.device}")

        try:
            gpu_tensor = test_tensor.cuda()
            print(f"   - GPUテンソルデバイス: {gpu_tensor.device}")
            print("✅ GPU動作テスト成功")
        except Exception as e:
            print(f"❌ GPU動作テスト失敗: {e}")
    else:
        print("❌ CUDA利用不可 - CPU版PyTorchまたはドライバー問題")

except ImportError as e:
    print(f"❌ PyTorchインポート失敗: {e}")

print("\n🔍 設定ファイルチェック...")
try:
    from config import settings

    print(f"✅ settings.py読み込み成功")
    print(f"   - USE_GPU: {settings.USE_GPU}")
    print(f"   - FORCE_GPU: {settings.FORCE_GPU}")
    print(f"   - GPU_DEVICE: {settings.GPU_DEVICE}")
    print(f"   - TORCH_DTYPE: {settings.TORCH_DTYPE}")
except Exception as e:
    print(f"❌ settings.py読み込み失敗: {e}")

print("\n🔍 LLMクラステスト...")
try:
    from src.core.llm import LLM

    print("✅ LLMクラス読み込み成功")

    # 小さなテストのため、モデル名を変更
    print("📦 モデルロードテスト開始（これには時間がかかります）...")
    llm = LLM(settings.MODEL)
    print("✅ LLMクラス初期化成功")

except Exception as e:
    print(f"❌ LLMクラステスト失敗: {e}")
    import traceback

    traceback.print_exc()
