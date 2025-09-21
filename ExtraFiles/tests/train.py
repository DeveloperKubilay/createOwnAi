import os
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast
)
from datasets import load_dataset

# === Tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(
    "../tokenizer",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>"
)

# === Config: Claude/GPT benzeri başlangıç ayarları ===
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=8192,
    n_ctx=8192,
    n_embd=768,
    n_layer=12,
    n_head=12,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# === Model ===
model = GPT2LMHeadModel(config)

# === Dataset ===
data = load_dataset("json", data_files="../all_css_sample_1M.jsonl", split="train")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=2048)

data = data.map(tokenize, batched=True, remove_columns=["text"])

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training ===
training_args = TrainingArguments(
    output_dir="../trained_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=False,
    save_safetensors=True,
    resume_from_checkpoint=True  # <-- CHECKPOINT'TEN DEVAM EDER
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# === Train time ===
trainer.train()

# === Save final model + tokenizer ===
model.save_pretrained("../trained_model")
tokenizer.save_pretrained("../trained_model")
