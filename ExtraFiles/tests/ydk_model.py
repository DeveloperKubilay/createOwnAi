import os
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast,
    AutoTokenizer,
    PreTrainedTokenizerFast
)
from datasets import load_dataset
import gc

# === GPU OPTIMIZE ===
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# === DEVICE CHECK ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Device: {device}")
if torch.cuda.is_available():
    print(f"💥 GPU: {torch.cuda.get_device_name()}")
    print(f"⚡ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# === Tokenizer ===
tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer")
tokenizer.add_special_tokens({
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
    'pad_token': '<pad>',
    'mask_token': '<mask>'
})

# === Config: Kaliteli model, optimize kod ===
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=8192,  # 8K token geri geldi
    n_ctx=8192,
    n_embd=768,  # Büyük model geri geldi
    n_layer=12,  # Full layers geri geldi
    n_head=12,   # Full heads geri geldi
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    activation_function="gelu_new",  # Modern activation
    attention_dropout=0.1,
    resid_dropout=0.1,
    use_cache=False  # Training için kapalı
)

# === Model ===
model = GPT2LMHeadModel(config)
model = model.to(device)
print(f"🧠 Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# === Memory AGGRESSIVE cleanup ===
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# === Dataset ===
print("📂 Tokenized data yükleniyor...")
data = load_dataset("json", data_files="./model_tokenized.jsonl", split="train")

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training: T4 OPTIMIZE + KALITE ===
training_args = TrainingArguments(
    output_dir="./trained_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # T4 VRAM için zorunlu
    gradient_accumulation_steps=16,   # Effective batch = 16
    save_steps=500,
    save_total_limit=3,
    prediction_loss_only=True,
    logging_steps=50,
    eval_steps=500,
    learning_rate=5e-5,  # Kalite için stable LR
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # T4 performance
    fp16_opt_level="O1",
    dataloader_pin_memory=False,  # Memory save
    dataloader_num_workers=0,  # Memory save
    remove_unused_columns=False,
    save_safetensors=True,
    resume_from_checkpoint=True,
    gradient_checkpointing=True,  # Memory zorunlu
    max_grad_norm=1.0,
    adam_epsilon=1e-8,  # Stable training
    lr_scheduler_type="cosine_with_restarts",
    optim="adamw_torch_fused",  # En hızlı optimizer
    ddp_find_unused_parameters=False,  # Speed boost
    torch_compile=False,  # T4 için kapalı
    report_to=[]  # Logging kapalı
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# === Train time ===
print("🚀 Training başlıyor...")
try:
    trainer.train()
    print("✅ Training tamamlandı!")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("💥 VRAM doldu! Batch size düşür")
        torch.cuda.empty_cache()
        gc.collect()
    raise e

# === Save final model + tokenizer ===
print("💾 Model kaydediliyor...")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("🎉 Hepsi tamam!")


#pip install torch transformers datasets accelerate