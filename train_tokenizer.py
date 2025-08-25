from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from itertools import islice
import orjson
import os
from multiprocessing import Pool, cpu_count
import mmap

os.environ["RAYON_NUM_THREADS"] = str(cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def process_chunk(chunk_data):
    lines, start_pos, end_pos = chunk_data
    texts = []
    for line in lines:
        try:
            data = orjson.loads(line.strip())
            if "content" in data and data["content"].strip():
                texts.append(data["content"])
        except:
            continue
    return texts

def iter_lines_parallel(path, batch_size=50000, num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    with open(path, "r", encoding="utf-8") as f:
        while True:
            lines = list(islice(f, batch_size))
            if not lines:
                break
            
            if len(lines) < num_workers:
                for line in lines:
                    try:
                        data = orjson.loads(line.strip())
                        if "content" in data and data["content"].strip():
                            yield data["content"]
                    except:
                        continue
            else:
                chunk_size = len(lines) // num_workers
                chunks = []
                for i in range(0, len(lines), chunk_size):
                    chunk = lines[i:i+chunk_size]
                    chunks.append((chunk, i, i+len(chunk)))
                
                with Pool(num_workers) as pool:
                    results = pool.map(process_chunk, chunks)
                
                for texts in results:
                    yield from texts

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

trainer = BpeTrainer(
    vocab_size=32000, 
    min_frequency=2,
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

print("🔥 Training başladı...")
tokenizer.train_from_iterator(iter_lines_parallel("./model.jsonl", batch_size=50000), trainer=trainer)
tokenizer.save("tokenizer")

#pip install tokenizers orjson
#pip install google-api-python-client google-auth-oauthlib tokenizers orjson