import os
import torch
import json
import shutil
from datetime import datetime
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

# === BACKUP SYSTEM ===
def create_backup_folder():
    """Backup klasörü oluştur"""
    backup_dir = "./backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    return backup_dir

def save_training_state(step, epoch, loss, model_path="/content/drive/Shareddrives/aidrive/trained_model"):
    """Training state kaydet"""
    backup_dir = create_backup_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    state = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "timestamp": timestamp,
        "model_path": model_path
    }

    # JSON state dosyası
    state_file = os.path.join(backup_dir, f"training_state_{timestamp}.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Son state'i de kaydet
    latest_state = os.path.join(backup_dir, "latest_state.json")
    with open(latest_state, "w") as f:
        json.dump(state, f, indent=2)

    print(f"💾 Backup kaydedildi: step {step}, epoch {epoch}, loss {loss:.4f}")
    return state_file

def load_latest_state():
    """Son kaydedilen state'i yükle"""
    latest_state = "./backups/latest_state.json"  # Local path kullan
    if os.path.exists(latest_state):
        with open(latest_state, "r") as f:
            state = json.load(f)
        print(f"🔄 Backup bulundu: step {state['step']}, epoch {state['epoch']}")
        return state
    return None

def backup_model(source_dir="/content/drive/Shareddrives/aidrive/trained_model", backup_name=None):
    """Model dosyalarını backup'la"""
    if not os.path.exists(source_dir):
        return

    backup_dir = create_backup_folder()
    if backup_name is None:
        backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backup_path = os.path.join(backup_dir, backup_name)

    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)

    shutil.copytree(source_dir, backup_path)
    print(f"📦 Model backup: {backup_path}")

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
tokenizer = PreTrainedTokenizerFast(tokenizer_file="/content/drive/Shareddrives/aidrive/tokenizer")
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
# Model'i checkpoint kontrolünden sonra yükleyeceğiz

# === Memory AGGRESSIVE cleanup ===
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# === Dataset ===
print("📂 Tokenized data yükleniyor...")
data = load_dataset("json", data_files="/content/drive/Shareddrives/aidrive/model_tokenized.jsonl", split="train")

# === Resume training check ===
resume_state = load_latest_state()
resume_from_checkpoint = None
model = None

# Checkpoint klasöründe gerçek checkpoint var mı kontrol et
checkpoint_files = []
model_dir = "/content/drive/Shareddrives/aidrive/trained_model"
if os.path.exists(model_dir):
    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith("checkpoint-")]

if checkpoint_files:
    # Tüm checkpoint'ları listele
    checkpoint_nums = [int(f.split("-")[1]) for f in checkpoint_files if f.split("-")[1].isdigit()]
    checkpoint_nums.sort()
    print(f"📁 Bulunan checkpoint'lar: {checkpoint_nums}")
    
    if checkpoint_nums:
        latest_checkpoint = max(checkpoint_nums)
        resume_from_checkpoint = f"{model_dir}/checkpoint-{latest_checkpoint}"
        print(f"🔄 En son checkpoint kullanılacak: checkpoint-{latest_checkpoint}")
        
        # Backup state ile checkpoint uyumsuzsa düzelt
        if resume_state:
            if resume_state['step'] < latest_checkpoint:
                print(f"⚠️ Backup state eski: {resume_state['step']}, checkpoint daha güncel: {latest_checkpoint}")
                print(f"✅ Checkpoint-{latest_checkpoint} kullanılacak (doğru davranış)")
            else:
                print(f"📊 Backup state: step {resume_state['step']}, checkpoint: {latest_checkpoint}")
        else:
            print(f"📊 Checkpoint-{latest_checkpoint}'dan devam ediliyor")
    
    # Model'i yine yeni oluştur, trainer checkpoint'tan yükleyecek
    model = GPT2LMHeadModel(config)
    
    # lm_head weight tie etme (missing key hatasını önlemek için)
    model.tie_weights()
    print("🧠 Yeni model oluşturuldu, checkpoint'tan weights yüklenecek")
    
elif resume_state and os.path.exists(f"{model_dir}/pytorch_model.bin"):
    # Sadece model dosyası var ama checkpoint yok
    print(f"🔄 Model dosyası bulundu: Step {resume_state['step']}")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        print("✅ Model dosyasından yüklendi")
    except:
        print("⚠️ Model yüklenemedi, yeni oluşturuluyor")
        model = GPT2LMHeadModel(config)
else:
    print("🆕 Sıfırdan başlıyor")
    model = GPT2LMHeadModel(config)

model = model.to(device)
print(f"🧠 Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# === Data collator ===
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training: CPU/GPU OPTIMIZE + KALITE ===
training_args = TrainingArguments(
    output_dir="/content/drive/Shareddrives/aidrive/trained_model",
    overwrite_output_dir=False,  # Checkpoint'ları korumak için False
    num_train_epochs=3,
    per_device_train_batch_size=2 if device.type == "cpu" else 1,  # CPU için biraz daha büyük
    gradient_accumulation_steps=8 if device.type == "cpu" else 16,   # CPU için daha az
    save_steps=500,
    save_total_limit=3,
    prediction_loss_only=True,
    logging_steps=50,
    eval_steps=500,
    learning_rate=5e-5,  # Kalite için stable LR
    warmup_steps=500,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Sadece GPU'da fp16
    fp16_opt_level="O1" if torch.cuda.is_available() else None,
    dataloader_pin_memory=torch.cuda.is_available(),  # Sadece GPU'da pin memory
    dataloader_num_workers=2 if device.type == "cpu" else 0,  # CPU için workers
    remove_unused_columns=False,
    save_safetensors=True,
    resume_from_checkpoint=resume_from_checkpoint,
    gradient_checkpointing=True,  # Memory zorunlu
    max_grad_norm=1.0,
    adam_epsilon=1e-8,  # Stable training
    lr_scheduler_type="cosine_with_restarts",
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # CPU için normal adamw
    ddp_find_unused_parameters=False,  # Speed boost
    torch_compile=False,  # Her ikisi için de kapalı
    report_to=[]  # Logging kapalı
)

class BackupTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup_interval = 500  # Her 500 step'te backup

    def log(self, logs, start_time=None):
        super().log(logs, start_time)

        # Her backup interval'da kaydet
        if self.state.global_step % self.backup_interval == 0 and self.state.global_step > 0:
            current_loss = logs.get("train_loss", 0.0)
            save_training_state(
                step=self.state.global_step,
                epoch=self.state.epoch,
                loss=current_loss
            )
            backup_model()

trainer = BackupTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# === Train time ===
print("🚀 Training başlıyor...")
try:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("✅ Training tamamlandı!")

    # Final backup
    save_training_state(
        step=trainer.state.global_step,
        epoch=trainer.state.epoch,
        loss=trainer.state.log_history[-1].get("train_loss", 0.0)
    )
    backup_model(backup_name="final_model")

except KeyboardInterrupt:
    print("⏸️ Training durduruldu, backup kaydediliyor...")
    save_training_state(
        step=trainer.state.global_step,
        epoch=trainer.state.epoch,
        loss=trainer.state.log_history[-1].get("train_loss", 0.0) if trainer.state.log_history else 0.0
    )
    backup_model(backup_name="interrupted_model")
    print("💾 Backup tamamlandı, kaldığı yerden devam edebilirsin!")

except RuntimeError as e:
    if "out of memory" in str(e):
        print("💥 VRAM doldu! Backup kaydediliyor...")
        save_training_state(
            step=trainer.state.global_step,
            epoch=trainer.state.epoch,
            loss=trainer.state.log_history[-1].get("train_loss", 0.0) if trainer.state.log_history else 0.0
        )
        backup_model(backup_name="oom_model")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    raise e

# === Save final model + tokenizer ===
print("💾 Model kaydediliyor...")
model.save_pretrained("/content/drive/Shareddrives/aidrive/trained_model")
tokenizer.save_pretrained("/content/drive/Shareddrives/aidrive/trained_model")

# Final cleanup - eski backup'ları temizle (son 5'i tut)
backup_dir = "./backups"  # Local backup dir kullan
if os.path.exists(backup_dir):
    state_files = [f for f in os.listdir(backup_dir) if f.startswith("training_state_") and f.endswith(".json")]
    state_files.sort(reverse=True)

    for old_file in state_files[5:]:  # Son 5'i tut
        os.remove(os.path.join(backup_dir, old_file))
        print(f"🗑️ Eski backup silindi: {old_file}")

print("🎉 Hepsi tamam!")


#pip install torch transformers datasets accelerate