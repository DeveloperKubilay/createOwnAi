from transformers import AutoTokenizer, AutoModelForCausalLM

# Tokenizer ve model yükleme
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")

# Prompt tanımlama
prompt = "app.get('/', "

# Tokenize etme
inputs = tokenizer(prompt, return_tensors="pt")

# Modeli çalıştırma
outputs = model.generate(**inputs)

# Çıktıyı çözümleme
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Çıktısı:", result)