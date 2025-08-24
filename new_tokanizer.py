
import sentencepiece as spm
import os
from multiprocessing import cpu_count

print("🔥 TXT'den direkt tokenizer train başlıyor...")
txt_path = "./model.txt"  # Buraya düz txt dosyanı koy, her satır bir örnek

spm.SentencePieceTrainer.train(
    input=txt_path,
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="bpe",
    num_threads=cpu_count(),
    train_extremely_large_corpus=True,
    character_coverage=0.9995,
    byte_fallback=True,
    split_digits=True,
    allow_whitespace_only_pieces=True,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<mask>"]
)

print("🔥 Tokenizer kaydedildi: tokenizer.model")