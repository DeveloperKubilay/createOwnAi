# === TPU için TensorFlow + HuggingFace ===
import tensorflow as tf
from transformers import (
    TFGPT2LMHeadModel,
    GPT2Config,
    GPT2TokenizerFast,
)
from datasets import load_dataset
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# === TPU cihazını bağla ===
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# === Tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(
    "../tokenizer",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>"
)

# === Config & Model ===
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

with strategy.scope():
    model = TFGPT2LMHeadModel(config)

# === Dataset ===
data = load_dataset("json", data_files="../all_css_sample_1M.jsonl", split="train")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=2048)

data = data.map(tokenize, batched=True, remove_columns=["text"])

# TF Dataset dönüştürme
def gen():
    for i in range(len(data)):
        yield ({'input_ids': data[i]['input_ids']}, data[i]['input_ids'])

tf_dataset = tf.data.Dataset.from_generator(
    gen,
    output_types=({'input_ids': tf.int32}, tf.int32),
    output_shapes=({'input_ids': (2048,)}, (2048,))
).shuffle(1000).batch(1)

# Mixed Precision Training'i etkinleştir
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Learning Rate Scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9
)

# === Compile & Train ===
with strategy.scope():
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn)

model.fit(tf_dataset, epochs=3)

# === Save ===
model.save_pretrained("../trained_model")
tokenizer.save_pretrained("../trained_model")
