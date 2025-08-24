import sentencepiece as spm
import os
import threading
from multiprocessing import cpu_count

print("🔥 Force multicore tokenizer başlıyor...")
txt_path = "./model.txt"
cores = cpu_count()

os.environ['OMP_NUM_THREADS'] = str(cores)
os.environ['MKL_NUM_THREADS'] = str(cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(cores)

print(f"💻 {cores} çekirdek zorla aktif edildi")

spm.SentencePieceTrainer.train(
    input=txt_path,
    model_prefix="tokenizer_multi",
    vocab_size=32000,
    model_type="bpe",
    num_threads=cores,
    train_extremely_large_corpus=True,
    character_coverage=0.9995,
    byte_fallback=True,
    split_digits=True,
    allow_whitespace_only_pieces=True,
    max_sentence_length=8192,
    input_sentence_size=5000000,
    shuffle_input_sentence=True,
    seed_sentencepiece_size=1000000,
    shrinking_factor=0.75,
    num_sub_iterations=2,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<mask>"]
)

print("🔥 Multi-core tokenizer kaydedildi: tokenizer_multi.model")
