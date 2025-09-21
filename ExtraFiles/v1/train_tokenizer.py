from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from itertools import islice
import orjson
import os
from multiprocessing import Pool, cpu_count
import mmap
from tqdm import tqdm

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

# Tokenizer var mı kontrol et
if os.path.exists("tokenizer"):
    print("✅ Tokenizer zaten var, atlıyor...")
else:
    print("🔥 Tokenizer training başladı...")
    tokenizer.train_from_iterator(iter_lines_parallel("./model.jsonl", batch_size=200000), trainer=trainer)
    tokenizer.save("tokenizer")
    print("✅ Tokenizer kaydedildi!")

# Tokenized data var mı kontrol et
if os.path.exists("model_tokenized.jsonl"):
    print("✅ Tokenized data zaten var, işlem tamamlandı!")
else:
    print("💾 Preprocessing başladı...")
    from transformers import PreTrainedTokenizerFast
    import json

    def tokenize_batch(texts):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer")
        fast_tokenizer.add_special_tokens({
            'bos_token': '<s>',
            'eos_token': '</s>',
            'unk_token': '<unk>',
            'pad_token': '<pad>',
            'mask_token': '<mask>'
        })
        
        results = []
        for text in texts:
            tokens = fast_tokenizer(
                text,
                truncation=True,
                max_length=8192,
                return_tensors=None
            )
            results.append(json.dumps({"input_ids": tokens["input_ids"]}) + "\n")
        return results

    batch_size = 50000  
    texts_batch = []
    
    with open("model_tokenized.jsonl", "w", encoding="utf-8") as out_file:
        for text in iter_lines_parallel("./model.jsonl", batch_size=200000):
            texts_batch.append(text)
            
            if len(texts_batch) >= batch_size:
                with Pool(cpu_count()) as pool:
                    chunk_size = len(texts_batch) // cpu_count()
                    batch_chunks = [texts_batch[i:i+chunk_size] for i in range(0, len(texts_batch), chunk_size)]
                    results = pool.map(tokenize_batch, batch_chunks)
                    
                    for chunk_results in results:
                        for line in chunk_results:
                            out_file.write(line)
                
                texts_batch = []
        
        # Son batch'i işle
        if texts_batch:
            with Pool(cpu_count()) as pool:
                chunk_size = len(texts_batch) // cpu_count() if len(texts_batch) > cpu_count() else 1
                batch_chunks = [texts_batch[i:i+chunk_size] for i in range(0, len(texts_batch), chunk_size)]
                if batch_chunks:
                    results = pool.map(tokenize_batch, batch_chunks)
                    
                    for chunk_results in results:
                        for line in chunk_results:
                            out_file.write(line)

    print("✅ Tokenized data kaydedildi!")

#pip install tokenizers orjson transformers