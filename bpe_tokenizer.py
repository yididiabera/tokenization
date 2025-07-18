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
    
    vocabulary = set(tokens)
    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(tokens)
        if not pair_freqs:
            break

        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)

        tokens = merge_most_frequent_pair(tokens, most_frequent_pair)

        vocabulary.add(''.join(most_frequent_pair))  # my learned vocabulary
    
    # Tokenization
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

# Example 
input_text = "I'm not in danger. I'm the danger!"
# input = "that 's far too tragic to merit such superficial treatment"
tokens = bpe_tokenizer(input_text)

print("Original text:", input_text)
print("BPE tokens:", tokens)