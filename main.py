from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🔥 Model yükleniyor...")

# Tokenizer ve model yükleme - doğru path
tokenizer = AutoTokenizer.from_pretrained("./trained_model")
model = AutoModelForCausalLM.from_pretrained("./trained_model")

print("✅ Model yüklendi!")
print(f"🧠 Model parametreleri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Test promptları
test_prompts = [
    "GPT",
    "Machine learning",
    "Python ile",
    "Training",
    "Neural network"
]

print("\n🚀 Model test ediliyor...")

for prompt in test_prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    
    # Tokenize etme
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Modeli çalıştırma - daha iyi parametreler
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,  # Kısa ama anlamlı
            temperature=0.8,  
            do_sample=True,
            top_p=0.9,  
            repetition_penalty=1.2,  # Tekrar önle
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Çıktıyı çözümleme - ULTRA clean decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # ByteLevel artifacts temizle
    generated_text = generated_text.replace("Ġ", " ")  # ByteLevel prefix
    generated_text = generated_text.replace("Ä±", "ı")  # Turkish i
    generated_text = generated_text.replace("ÅŁ", "ş")   # Turkish ş  
    generated_text = generated_text.replace("Ã§", "ç")   # Turkish ç
    generated_text = generated_text.replace("Ã¼", "ü")   # Turkish ü
    generated_text = generated_text.replace("Ã¶", "ö")   # Turkish ö
    generated_text = generated_text.replace("ÄŁ", "ğ")   # Turkish ğ
    
    # Extra spaces temizle
    import re
    generated_text = re.sub(r'\s+', ' ', generated_text).strip()
    
    # Sadece yeni üretilen kısmı al (prompt'u çıkar)
    new_text = generated_text[len(prompt):].strip()
    
    print(f"🤖 Model cevabı: '{prompt}{new_text}'")

print("\n🎉 Bitti!")