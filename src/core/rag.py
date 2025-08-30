import os
import time
from pathlib import Path
import faiss
import numpy as np


def file_exists(file_path):
    return os.path.isfile(file_path)


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def text_to_vector(text, vector_dim=512):
    # Fixed dimension vector representation
    # Simple hash-based approach for consistent dimensions
    vector = [0.0] * vector_dim
    for i, char in enumerate(text):
        if i >= vector_dim:
            break
        vector[i % vector_dim] += ord(char) / 1000.0  # Normalize
    return vector


class RAG:
    def __init__(self, vector_dim=512):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)
        self.texts = []  # Store original texts for reference

    def __del__(self):
        try:
            # Python終了時にモジュールが利用できない場合があるため、安全に処理
            if hasattr(self, 'index') and self.index is not None:
                # 自動保存は無効化（明示的にsave_index()を呼び出すことを推奨）
                pass
            if hasattr(self, 'texts') and self.texts:
                # 自動保存は無効化（明示的にsave_texts()を呼び出すことを推奨）
                pass
        except Exception:
            # デストラクタでの例外は無視
            pass

    def add_text(self, text):
        vector = text_to_vector(text, self.vector_dim)
        self.index.add(np.array([vector]).astype(np.float32))
        self.texts.append(text)

    def query(self, text, k=5):
        # Check if there are any texts in the index
        if len(self.texts) == 0:
            return []

        vector = text_to_vector(text, self.vector_dim)
        actual_k = min(k, len(self.texts))

        try:
            D, I = self.index.search(np.array([vector]).astype(np.float32), k=actual_k)
        except Exception as e:
            print(f"Search error: {e}")
            return []

        # Return actual text content and distances
        results = []
        for i, idx in enumerate(I[0]):
            if idx != -1 and idx < len(self.texts):
                results.append(
                    {
                        "text": self.texts[idx][:200] + "..." if len(self.texts[idx]) > 200 else self.texts[idx],
                        "distance": D[0][i],
                        "index": idx,
                    }
                )
        return results

    def add_log_file(self, file_path):
        """ログファイル専用の追加メソッド"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # ログファイルを行ごとに分割して追加
            lines = content.split("\n")
            for line in lines:
                if line.strip():  # 空行でない場合のみ追加
                    self.add_text(line.strip())

            print(f"ログファイル '{file_path}' から {len([l for l in lines if l.strip()])} 行を追加しました。")
            return True

        except Exception as e:
            print(f"ログファイル読み込みエラー: {e}")
            return False

    def search_by_keyword(self, keyword, k=10):
        """キーワードでログエントリを検索"""
        results = self.query(keyword, k=k)
        return results

    def add_file(self, file_path):
        text = load_text_file(file_path)
        self.add_text(text)

    def add_directory(self, dir_path):
        for file_path in Path(dir_path).rglob("*.txt"):
            self.add_file(file_path)

    @property
    def vector_store(self):
        try:
            return self.index
        except Exception as e:
            print(f"Error accessing vector store: {e}")

    def save_all(self, base_path="./rag_data"):
        """インデックスとテキストを一括保存"""
        try:
            import time
            now = time.strftime("%Y%m%d_%H%M%S")
            
            if self.index is not None and len(self.texts) > 0:
                index_path = f"{base_path}/{now}/index.faiss"
                texts_path = f"{base_path}/{now}/texts.txt"
                
                self.save_index(index_path)
                self.save_texts(texts_path)
                
                print(f"データを保存しました: {base_path}/{now}/")
                return True
            else:
                print("保存するデータがありません")
                return False
                
        except Exception as e:
            print(f"保存エラー: {e}")
            return False

    def save_index(self, file_path):
        try:
            mkdir_p(os.path.dirname(file_path))
            faiss.write_index(self.index, file_path)
            print(f"Index saved to {file_path}")
            return True
        except Exception as e:
            print(f"インデックス保存エラー: {e}")
            return False

    def load_index(self, file_path):
        if not file_exists(file_path):
            print(f"File {file_path} does not exist")
            return
        self.index = faiss.read_index(file_path)
        print(f"Index loaded from {file_path}")

    def save_texts(self, file_path):
        try:
            mkdir_p(os.path.dirname(file_path))
            with open(file_path, "w", encoding="utf-8") as file:
                for text in self.texts:
                    file.write(text + "\n")
            print(f"Texts saved to {file_path}")
            return True
        except Exception as e:
            print(f"テキスト保存エラー: {e}")
            return False


if __name__ == "__main__":
    rag = RAG()
    # rag.load_index("path/to/index/file")
    rag.add_file("./requirements.txt")

    # Add more sample texts for better search results
    rag.add_text("Python is a programming language")
    rag.add_text("Requirements file contains package dependencies")
    rag.add_text("Machine learning and data science tools")

    print("Vector store:", rag.vector_store)
    print("\nSearch results:")
    results = rag.query("python requirements")
    for i, result in enumerate(results):
        print(f"{i+1}. Distance: {result['distance']:.2f}")
        print(f"   Text: {result['text']}")
        print()

    del rag
