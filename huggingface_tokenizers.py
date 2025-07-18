from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

# Load tokenizers
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load first 20 samples from SST-2 dataset
dataset = load_dataset("glue", "sst2", split="train[:20]")

# Show outputs
# for i, sample in enumerate(dataset):
#     text = sample["sentence"]
#     bert_tokens = bert_tokenizer.tokenize(text)
#     gpt2_tokens = gpt2_tokenizer.tokenize(text)
    
#     print(f"\nSample {i+1}")
#     print(f"Sentence: {text}")
#     print(f"BERT tokens: {bert_tokens}")
#     print(f"GPT-2 tokens: {gpt2_tokens}")

sentence = "i'm not in danger. i'm the danger!"

bert_tokens = bert_tokenizer.tokenize(sentence)
gpt2_tokens = gpt2_tokenizer.tokenize(sentence)

print("BERT Tokens:", bert_tokens)
print("GPT-2 Tokens:", gpt2_tokens)