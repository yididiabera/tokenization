import torch
import torch.nn as nn
from collections import Counter

def whitespace_tokenizer(text):
    "A basic whitespace tokenizer that splits text into tokens based on whitespace."
    
    # Split the text on any whitespace character (space, tab, newline)
    tokens = text.split()
    
    return tokens

def wordLevel_tokenizer(text):
    " Tokenizes text into words, handles punctuation, and splits contractions."
    # adding space before apostrophes to split contractions 
    text = text.replace("'", " '")

    # adding spaces around punctuation
    punctuations = ".,!?()"
    for p in punctuations:
        text = text.replace(p, f" {p} ")

    # replacing multiple spaces with a single space
    text = text.split()
    text = ' '.join(text)

    # splitting into tokens
    tokens = text.strip().split()

    return tokens

def bpe_tokenizer(text, num_merges=10):
    " A simple Byte Pair Encoding (BPE) tokenizer that merges frequent character pairs."
    # Step 1: Initialize with character-level tokens
    tokens = list(text)
    
    # Helper function to get pair frequencies
    def get_pair_frequencies(tokens):
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    # Helper function to merge the most frequent pair
    def merge_most_frequent_pair(tokens, pair):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    # Step 2: Create vocabulary by merging frequent pairs
    vocabulary = set(tokens)
    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(tokens)
        if not pair_freqs:
            break
        # Find the most frequent pair
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
        # Merge the most frequent pair
        tokens = merge_most_frequent_pair(tokens, most_frequent_pair)
        # Add merged pair to vocabulary
        vocabulary.add(''.join(most_frequent_pair))
    
    # Step 3: Tokenize using the learned vocabulary
    final_tokens = []
    i = 0
    while i < len(text):
        for j in range(len(text), i, -1):
            candidate = text[i:j]
            if candidate in vocabulary:
                final_tokens.append(candidate)
                i = j
                break
        else:
            final_tokens.append(text[i])
            i += 1
    
    return final_tokens


# 2. Vocabulary and Numericalization
class Vocabulary:
    def __init__(self, tokens_list, min_freq=1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        counter = Counter()
        for tokens in tokens_list:
            counter.update(tokens)
            
        for word, freq in counter.items():
            if freq >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def numericalize(self, tokens):
        return [self.word2idx.get(token, 1) for token in tokens]  # 1=UNK

# 3. PyTorch Embedding Wrapper
class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Index for <PAD>
        )
        # Initialize weights
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, token_ids):
        # token_ids: List of numericalized tokens
        return self.embedding(torch.tensor(token_ids))

# 4. Full Workflow Example
def generate_embeddings(text, tokenizer_type='word'):
    # Choose tokenizer
    if tokenizer_type == 'whitespace':
        tokens = whitespace_tokenizer(text.lower())
    elif tokenizer_type == 'word':
        tokens = wordLevel_tokenizer(text.lower())
    elif tokenizer_type == 'bpe':
        tokens = bpe_tokenizer(text.lower())
    else:
        raise ValueError("Invalid tokenizer type")
    
    # Build vocabulary (in practice, you'd pre-build this on your full dataset)
    vocab = Vocabulary([tokens])
    
    # Numericalize tokens
    token_ids = vocab.numericalize(tokens)
    
    # Create embedding layer
    embedder = TokenEmbedder(len(vocab.word2idx))
    
    # Get embeddings
    with torch.no_grad():  # No training needed for this example
        embeddings = embedder(token_ids)
    
    return tokens, embeddings, vocab

# Example usage
text = "I'm not in danger. I'm the danger!"

# Compare different tokenizers
for tokenizer in ['whitespace', 'word', 'bpe']:
    tokens, embeddings, vocab = generate_embeddings(text, tokenizer)
    
    print(f"\n{tokenizer.upper()} Tokenizer:")
    print("Tokens:", tokens)
    print("Vocabulary size:", len(vocab.word2idx))
    print("Embedding shape:", embeddings.shape)
    print("Sample embedding for first token:")
    print(embeddings[0][:10])  