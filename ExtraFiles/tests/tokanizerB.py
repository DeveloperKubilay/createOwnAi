from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from itertools import islice
import orjson

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

def iter_lines(path, batch_size):
    with open(path, "r", encoding="utf-8") as f:
        while True:
            lines = list(islice(f, batch_size))
            if not lines:
                break
            for line in lines:
                data = orjson.loads(line.strip())
                if "content" in data:
                    yield data["content"]

trainer = BpeTrainer(vocab_size=32000, min_frequency=3,
                     special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.train_from_iterator(iter_lines("./model.jsonl", batch_size=20000), trainer=trainer)
tokenizer.save("tokenizer")

#pip install tokenizers orjson