import json
import os
import sys
from tqdm import tqdm

# Komut satırından dosya adlarını al
if len(sys.argv) < 3:
    print("Usage: python clean_jsonl.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
BUFFER_SIZE = 50000 # Chunk boyutu

# Eğer output_file varsa sil
if os.path.exists(output_file):
    os.remove(output_file)

buffer = []

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Cleaning JSONL file"):
        try:
            data = json.loads(line.strip())
            if "response" in data and "question" in data:
                new_data = {"content": f"{data['question']} {data['response']}"}
                buffer.append(json.dumps(new_data, ensure_ascii=False))
                # Var olan datayı düzenle ve ekle
                del data["question"]
                del data["response"]
                buffer.append(json.dumps(data, ensure_ascii=False))
            elif "content" in data:
                # Sadece content olanları ekle
                buffer.append(json.dumps(data, ensure_ascii=False))

            # Buffer dolduğunda yaz
            if len(buffer) >= BUFFER_SIZE:
                outfile.write("\n".join(buffer) + "\n")
                buffer = []
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line.strip()}")

    # Kalan buffer'ı yaz
    if buffer:
        outfile.write("\n".join(buffer) + "\n")
