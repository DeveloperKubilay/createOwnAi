import json
import sys
from tqdm import tqdm

# Komut satırından dosya adı al
if len(sys.argv) < 2:
    print("Usage: python count_parameters.py <input_file> [-length]")
    sys.exit(1)

input_file = sys.argv[1]
count_length = "-length" in sys.argv

# JSONL dosyasındaki parametreleri say veya toplam JSON sayısını hesapla
if count_length:
    total_lines = 0
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="Counting total JSONs"):
            try:
                json.loads(line.strip())
                total_lines += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    print(f"Total JSON objects: {total_lines}")
else:
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="Checking keys outside 'content'"):
            try:
                data = json.loads(line.strip())
                extra_keys = [key for key in data.keys() if key != "content"]
                if extra_keys:
                    print(f"Line has extra keys: {extra_keys}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
