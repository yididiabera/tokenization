def character_tokenizer(text):
    " A character-level tokenizer that splits text into individual characters."
    token = list(text)
    
    return token

# Example 
input = "I'm not in danger. I'm the danger!"
# input = "that 's far too tragic to merit such superficial treatment."
tokens = character_tokenizer(input)

print("Original text:", input)
print("Character tokens:", tokens)