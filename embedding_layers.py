import torch
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


def whitespace_tokenizer(text):
    "A basic whitespace tokenizer that splits text into tokens based on whitespace."
    
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
    tokens = list(text)
    
    def get_pair_frequencies(tokens):
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
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
    
    # Create vocabulary by merging frequent pairs
    vocabulary = set(tokens)
    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(tokens)
        if not pair_freqs:
            break
        
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)

        tokens = merge_most_frequent_pair(tokens, most_frequent_pair)

        vocabulary.add(''.join(most_frequent_pair))
    
    # tokenization 
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

class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0 
        )
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        
    def forward(self, token_ids):
        return self.embedding(torch.tensor(token_ids))

def generate_embeddings(text, tokenizer_type='word'):
    if tokenizer_type == 'whitespace':
        tokens = whitespace_tokenizer(text.lower())
    elif tokenizer_type == 'word':
        tokens = wordLevel_tokenizer(text.lower())
    elif tokenizer_type == 'bpe':
        tokens = bpe_tokenizer(text.lower())
    else:
        raise ValueError("Invalid tokenizer type")
    
    vocab = Vocabulary([tokens])
    
    token_ids = vocab.numericalize(tokens)
    
    embedder = TokenEmbedder(len(vocab.word2idx))
    
    with torch.no_grad():  
        embeddings = embedder(token_ids)
    
    return tokens, embeddings, vocab


# Example usage
text = "I'm not in danger. I'm the danger!"

for tokenizer in ['whitespace', 'word', 'bpe']:
    tokens, embeddings, vocab = generate_embeddings(text, tokenizer)
    
    print(f"\n{tokenizer.upper()} Tokenizer:")
    print("Tokens:", tokens)
    print("Vocabulary size:", len(vocab.word2idx))
    print("Embedding shape:", embeddings.shape)
    print("Sample embedding for first token:")
    print(embeddings[0][:10])  


def visualize_embeddings(all_tokens, all_embeddings, all_labels):
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0).numpy()
    reduced_embeddings = PCA(n_components=2).fit_transform(all_embeddings_tensor)

    plt.figure(figsize=(10, 6))
    for tokenizer_type in set(all_labels):
        indices = [i for i, label in enumerate(all_labels) if label == tokenizer_type]
        x = [reduced_embeddings[i][0] for i in indices]
        y = [reduced_embeddings[i][1] for i in indices]
        plt.scatter(x, y, label=tokenizer_type.upper(), s=60)

        for i in indices:
            plt.text(reduced_embeddings[i][0], reduced_embeddings[i][1], all_tokens[i], fontsize=9)

    plt.title("Token Embeddings Visualized with PCA")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("embedding_visualization.png", dpi=300)
    print("✅ Plot saved as 'embedding_visualization.png'")


all_tokens = []
all_embeddings = []
all_labels = []

for tokenizer in ['whitespace', 'word', 'bpe']:
    tokens, embeddings, vocab = generate_embeddings(text, tokenizer)
    all_tokens.extend(tokens)
    all_embeddings.append(embeddings)
    all_labels.extend([tokenizer] * len(tokens))

visualize_embeddings(all_tokens, all_embeddings, all_labels)
