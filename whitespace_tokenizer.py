def whitespace_tokenizer(text):
    "A basic whitespace tokenizer that splits text into tokens based on whitespace."
    
    # Split the text on any whitespace character (space, tab, newline)
    tokens = text.split()
    
    return tokens

# Example 
input = "I am not in danger. I am the danger!"
tokens = whitespace_tokenizer(input)

print("Original text:", input)
print("Tokens:", tokens)