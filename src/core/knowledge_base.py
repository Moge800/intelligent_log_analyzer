import csv
import os


class KnowledgeBase:
    """問題対策のナレッジベースを管理するクラス"""

    def __init__(self, csv_file_path="knowledge_base.csv"):
        """ナレッジベースを初期化"""
        self.csv_file_path = csv_file_path
        self.knowledge_data = []
        self.headers = []
        self.load_csv()

    def load_csv(self):
        """CSVファイルからナレッジベースを読み込み"""
        try:
            if not os.path.exists(self.csv_file_path):
                print(f"ナレッジベースファイルが見つかりません: {self.csv_file_path}")
                return False

            with open(self.csv_file_path, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                self.headers = next(csv_reader)  # ヘッダー行を取得

                for row in csv_reader:
                    if len(row) >= len(self.headers):  # 行に十分なデータがある場合
                        knowledge_entry = {}
                        for i, header in enumerate(self.headers):
                            knowledge_entry[header] = row[i] if i < len(row) else ""
                        self.knowledge_data.append(knowledge_entry)

            print(f"ナレッジベース読み込み完了: {len(self.knowledge_data)}件のエントリ")
            return True

        except Exception as e:
            print(f"CSVファイル読み込みエラー: {e}")
            return False

    def search_solutions(self, keywords, category=None):
        """キーワードとカテゴリで解決策を検索"""
        results = []
        search_keywords = [keyword.lower() for keyword in keywords]

        for entry in self.knowledge_data:
            # カテゴリフィルタ
            if category and entry.get("カテゴリ", "").lower() != category.lower():
                continue

            # キーワード検索
            match_score = 0
            entry_text = " ".join(entry.values()).lower()

            for keyword in search_keywords:
                if keyword in entry_text:
                    match_score += 1

            if match_score > 0:
                entry_with_score = entry.copy()
                entry_with_score["match_score"] = match_score
                results.append(entry_with_score)

        # マッチスコアでソート
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    def get_solution_by_name(self, problem_name):
        """問題名で完全一致検索"""
        for entry in self.knowledge_data:
            if entry.get("問題名", "").lower() == problem_name.lower():
                return entry
        return None

    def get_all_categories(self):
        """すべてのカテゴリを取得"""
        categories = set()
        for entry in self.knowledge_data:
            if entry.get("カテゴリ"):
                categories.add(entry["カテゴリ"])
        return sorted(list(categories))

    def format_solution(self, solution_entry):
        """解決策を読みやすい形式でフォーマット"""
        if not solution_entry:
            return "該当する解決策が見つかりませんでした。"

        formatted = f"""
【問題】: {solution_entry.get('問題名', 'N/A')}
【カテゴリ】: {solution_entry.get('カテゴリ', 'N/A')}
【対処法】: {solution_entry.get('対処法', 'N/A')}
【詳細説明】: {solution_entry.get('詳細説明', 'N/A')}
【予防策】: {solution_entry.get('予防策', 'N/A')}
【参考情報】: {solution_entry.get('参考情報', 'N/A')}
"""
        return formatted.strip()

    def add_knowledge_entry(self, problem_name, category, solution, details="", prevention="", reference=""):
        """新しいナレッジエントリを追加"""
        new_entry = {
            "問題名": problem_name,
            "カテゴリ": category,
            "対処法": solution,
            "詳細説明": details,
            "予防策": prevention,
            "参考情報": reference,
        }
        self.knowledge_data.append(new_entry)
        return True

    def save_to_csv(self, output_path=None):
        """ナレッジベースをCSVファイルに保存"""
        if output_path is None:
            output_path = self.csv_file_path

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

                for entry in self.knowledge_data:
                    row = [entry.get(header, "") for header in self.headers]
                    writer.writerow(row)

            print(f"ナレッジベースを保存しました: {output_path}")
            return True

        except Exception as e:
            print(f"CSV保存エラー: {e}")
            return False


def main():
    """ナレッジベースのテスト"""
    kb = KnowledgeBase("knowledge_base.csv")

    # カテゴリ一覧表示
    print("利用可能なカテゴリ:")
    for category in kb.get_all_categories():
        print(f"  - {category}")

    # キーワード検索テスト
    print("\n=== キーワード検索テスト ===")
    results = kb.search_solutions(["データベース", "接続"])
    for result in results[:2]:  # 上位2件を表示
        print(kb.format_solution(result))
        print("-" * 50)

    # 問題名での完全一致検索
    print("\n=== 完全一致検索テスト ===")
    solution = kb.get_solution_by_name("メモリ使用量超過")
    print(kb.format_solution(solution))


if __name__ == "__main__":
    main()
