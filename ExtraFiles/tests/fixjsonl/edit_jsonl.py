import json
from tqdm import tqdm

# JSONL dosyasını düzenle
input_file = "test.jsonl"
output_file = "edited_test.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Editing JSONL file"):
        try:
            data = json.loads(line.strip())
            if "question" in data and "response" in data:
                # Yeni content alanı oluştur
                data["content"] = f"{data['question']} {data['response']}"
                # question ve response alanlarını sil
                del data["question"]
                del data["response"]
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")
