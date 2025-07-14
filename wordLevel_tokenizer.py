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

# Example 
input = "I'm not in danger. I'm the danger!"
tokens = wordLevel_tokenizer(input)

print("Original text:", input)
print("Tokens:", tokens)