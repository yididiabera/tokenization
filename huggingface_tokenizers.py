from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

# Load the first 30 samples from the SST-2 dataset
dataset = load_dataset("sst2")["train"].select(range(30))
corpus = [sample["sentence"] for sample in dataset]

# Initialize Hugging Face tokenizers
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt-2")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Function to tokenize and decode tokens for display
def tokenize_and_display(text, tokenizer, name):
    tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens)
    return decoded_tokens

# Select a sample sentence for demonstration (first sentence from corpus)
sample_sentence = corpus[0]
print(f"Sample Sentence: {sample_sentence}\n")

# Dictionary to store results
results = {
    "Tokenizer": [],
    "Tokens": []
}

# Tokenize with each tokenizer
for tokenizer, name in [(bert_tokenizer, "BERT"), (gpt2_tokenizer, "GPT-2"), (t5_tokenizer, "T5")]:
    tokens = tokenize_and_display(sample_sentence, tokenizer, name)
    results["Tokenizer"].append(name)
    results["Tokens"].append(tokens)

# Create a DataFrame for side-by-side comparison
df = pd.DataFrame(results)

# Display results
print("Tokenizer Outputs for Sample Sentence:")
print(df.to_string(index=False))

# Optionally, save to CSV for further inspection
df.to_csv("tokenizer_comparison.csv", index=False)

# Example: Tokenize the entire corpus (first 5 sentences for brevity)
print("\nTokenizing first 5 sentences of the corpus:")
for i, sentence in enumerate(corpus[:5]):
    print(f"\nSentence {i+1}: {sentence}")
    for tokenizer, name in [(bert_tokenizer, "BERT"), (gpt2_tokenizer, "GPT-2"), (t5_tokenizer, "T5")]:
        tokens = tokenize_and_display(sentence, tokenizer, name)
        print(f"{name}: {tokens}")

# from transformers import BertTokenizer, GPT2Tokenizer
# from datasets import load_dataset

# # Load tokenizers
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# # Load first 20 samples from SST-2 dataset
# dataset = load_dataset("glue", "sst2", split="train[:20]")

# # Show outputs
# for i, sample in enumerate(dataset):
#     text = sample["sentence"]
#     bert_tokens = bert_tokenizer.tokenize(text)
#     gpt2_tokens = gpt2_tokenizer.tokenize(text)
    
#     print(f"\nSample {i+1}")
#     print(f"Sentence: {text}")
#     print(f"BERT tokens: {bert_tokens}")
#     print(f"GPT-2 tokens: {gpt2_tokens}")
