from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import orjson
import os
import gc
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import json

# Multiprocessing optimize
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def process_lines_batch(lines_batch):
    """Batch'i process et - multiprocessing için"""
    texts = []
    for line in lines_batch:
        try:
            data = orjson.loads(line.strip())
            if "content" in data and data["content"].strip():
                texts.append(data["content"])
        except:
            continue
    return texts

def iter_lines_optimized(path, batch_size=10000):
    """Optimize iterator - büyük dosyalar için"""
    print(f"📂 {path} dosyası okunuyor...")
    
    # Dosya boyutunu kontrol et
    file_size = os.path.getsize(path)
    print(f"📏 Dosya boyutu: {file_size / (1024*1024):.1f}MB")
    
    total_lines = sum(1 for _ in open(path, 'r', encoding='utf-8'))
    print(f"📖 Toplam satır: {total_lines}")
    
    if total_lines < 1000:
        # Küçük dosya - basit yöntem
        print("🚀 Küçük dosya, basit işlem...")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = orjson.loads(line.strip())
                    if "content" in data and data["content"].strip():
                        count += 1
                        yield data["content"]
                except:
                    continue
        print(f"✅ {count} satır işlendi!")
    else:
        # Büyük dosya - multiprocessing
        print("💥 Büyük dosya, multiprocessing aktif...")
        count = 0
        processed = 0
        
        with open(path, "r", encoding="utf-8") as f:
            lines_batch = []
            
            for line in f:
                lines_batch.append(line)
                
                if len(lines_batch) >= batch_size:
                    # Process batch
                    num_workers = min(cpu_count(), 4)  # Max 4 worker
                    chunk_size = len(lines_batch) // num_workers
                    
                    if chunk_size > 0:
                        chunks = [lines_batch[i:i+chunk_size] for i in range(0, len(lines_batch), chunk_size)]
                        
                        with ProcessPoolExecutor(max_workers=num_workers) as executor:
                            results = list(executor.map(process_lines_batch, chunks))
                            
                        for texts in results:
                            for text in texts:
                                count += 1
                                yield text
                    
                    processed += len(lines_batch)
                    if processed % 50000 == 0:
                        print(f"⚡ {processed} satır işlendi, {count} text bulundu...")
                    
                    lines_batch = []
                    gc.collect()  # Memory temizle
            
            # Son batch
            if lines_batch:
                num_workers = min(cpu_count(), len(lines_batch) // 100 + 1)
                chunk_size = len(lines_batch) // num_workers if num_workers > 0 else len(lines_batch)
                
                if chunk_size > 0:
                    chunks = [lines_batch[i:i+chunk_size] for i in range(0, len(lines_batch), chunk_size)]
                    
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        results = list(executor.map(process_lines_batch, chunks))
                        
                    for texts in results:
                        for text in texts:
                            count += 1
                            yield text
        
        print(f"🎉 Toplam {count} text işlendi!")

tokenizer = Tokenizer(BPE())
# ByteLevel yerine Whitespace pre-tokenizer kullan
from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8000,  # Daha küçük vocab (büyük dataset yok)
    min_frequency=1,  # Daha düşük min freq
    show_progress=True,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# Tokenizer var mı kontrol et
if os.path.exists("tokenizer"):
    print("✅ Tokenizer zaten var, atlıyor...")
else:
    print("🔥 Tokenizer training başladı...")
    tokenizer.train_from_iterator(iter_lines_optimized("./model.jsonl"), trainer=trainer)
    tokenizer.save("tokenizer")
    print("✅ Tokenizer kaydedildi!")

# Tokenized data var mı kontrol et
if os.path.exists("model_tokenized.jsonl"):
    print("✅ Tokenized data zaten var, işlem tamamlandı!")
else:
    print("💾 Preprocessing başladı...")
    from transformers import PreTrainedTokenizerFast

    def tokenize_batch_optimized(texts_chunk):
        """Tokenize batch - multiprocessing için optimize"""
        try:
            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer")
            fast_tokenizer.add_special_tokens({
                'bos_token': '<s>',
                'eos_token': '</s>',
                'unk_token': '<unk>',
                'pad_token': '<pad>',
                'mask_token': '<mask>'
            })
            
            results = []
            for text in texts_chunk:
                try:
                    tokens = fast_tokenizer(
                        text,
                        truncation=True,
                        max_length=8192,
                        return_tensors=None
                    )
                    results.append(json.dumps({"input_ids": tokens["input_ids"]}) + "\n")
                except:
                    continue
            return results
        except Exception as e:
            print(f"⚠️ Batch tokenize hatası: {e}")
            return []

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer")
    fast_tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'mask_token': '<mask>'
    })
    
    count = 0
    batch_size = 5000  # Küçük batch'ler
    texts_batch = []
    
    with open("model_tokenized.jsonl", "w", encoding="utf-8") as out_file:
        for text in iter_lines_optimized("./model.jsonl"):
            texts_batch.append(text)
            
            if len(texts_batch) >= batch_size:
                # Multiprocessing ile tokenize
                num_workers = min(cpu_count(), 4)
                chunk_size = len(texts_batch) // num_workers
                
                if chunk_size > 0 and len(texts_batch) > 100:
                    # Büyük batch - multiprocessing
                    chunks = [texts_batch[i:i+chunk_size] for i in range(0, len(texts_batch), chunk_size)]
                    
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        results = list(executor.map(tokenize_batch_optimized, chunks))
                        
                    for chunk_results in results:
                        for line in chunk_results:
                            out_file.write(line)
                            count += 1
                else:
                    # Küçük batch - basit
                    for text in texts_batch:
                        try:
                            tokens = fast_tokenizer(
                                text,
                                truncation=True,
                                max_length=8192,
                                return_tensors=None
                            )
                            out_file.write(json.dumps({"input_ids": tokens["input_ids"]}) + "\n")
                            count += 1
                        except:
                            continue
                
                if count % 1000 == 0:
                    print(f"🔢 {count} text tokenize edildi...")
                
                texts_batch = []
                gc.collect()
        
        # Son batch
        if texts_batch:
            for text in texts_batch:
                try:
                    tokens = fast_tokenizer(
                        text,
                        truncation=True,
                        max_length=8192,
                        return_tensors=None
                    )
                    out_file.write(json.dumps({"input_ids": tokens["input_ids"]}) + "\n")
                    count += 1
                except:
                    continue

    print(f"✅ {count} tokenized data kaydedildi!")

#pip install tokenizers orjson transformers