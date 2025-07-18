# Tokenization and Embeddings Comparison Project

## Project Overview
This project implements and compares different tokenization strategies, ranging from basic custom methods to advanced pretrained models. It includes visualization of embedding representations using dimensionality reduction techniques like PCA.

## Features

### Implemented Tokenizers
1. **Character-level**: Splits text into individual characters.
2. **Whitespace**: Tokenizes based on spaces.
3. **Word-level**: Handles punctuation and contractions more intelligently.
4. **BPE (Byte Pair Encoding)**: Simplified implementation to learn subword units.
5. **Hugging Face Tokenizers**: Uses pretrained models like **BERT** and **GPT-2** for advanced tokenization.

### Core Functionality
- Vocabulary construction and token-to-index mapping.
- Embedding generation using PyTorch’s `nn.Embedding`.
- 2D visualization of embedding spaces using PCA for interpretation.
- Side-by-side comparison of different tokenizers on the same input text.

## File Structure



# Tokenization and Embeddings Comparison Project

## Project Overview
This project implements and compares different tokenization strategies, from basic to advanced, and visualizes their embedding representations. The implementation includes custom tokenizers and Hugging Face's pretrained tokenizers.

## Features

### Implemented Tokenizers
1. **Character-level**: Splits text into individual characters
2. **Whitespace**: Basic space-based tokenization
3. **Word-level**: Handles punctuation and contractions
4. **BPE**: Simple Byte Pair Encoding implementation
5. **Hugging Face** (BERT, GPT-2): Professional tokenizers for comparison

### Core Functionality
- Vocabulary building and numericalization
- Embedding generation with PyTorch
- 2D visualization of embeddings (PCA)
- Side-by-side tokenizer comparisons

## File Structure
```
tokenization/
├── character_tokenizer.py    # Character-level implementation
├── whitespace_tokenizer.py   # Space-based tokenizer
├── wordLevel_tokenizer.py    # Word-level with punctuation handling
├── bpe_tokenizer.py          # Custom BPE implementation
├── embedding_layers.py       # PyTorch embedding generation
├── huggingface_tokenizers.py # BERT/GPT-2 comparison
└── visualization.py          # Embedding visualization
```

## Usage

### Running Tokenizers
```bash
# Test individual tokenizers
python3 character_tokenizer.py
python3 whitespace_tokenizer.py
python3 wordLevel_tokenizer.py
python3 bpe_tokenizer.py

# Compare Hugging Face tokenizers
python3 huggingface_tokenizers.py
```

### Generating and Visualizing Embeddings
```bash
python3 embedding_layers.py
```

### Expected Output
Each script will display:
1. Original text
2. Tokenized output
3. Vocabulary information
4. Sample embeddings (for embedding scripts)
5. Visualization plot (when applicable)

## Key Findings
- **Granularity Spectrum**: Character → Whitespace → Word → BPE → HF Tokenizers
- **Punctuation Handling**: Major differences in treatment of symbols
- **Space Representation**: Varies from explicit tokens to prefix markers
- **Vocabulary Size**: Character < Whitespace < Word < BPE < HF

## Dependencies
- Python 3.10+
- Required packages:
  ```bash
  pip install torch transformers datasets scikit-learn matplotlib
  ```

## Author
Jedidiah - NLP Tokenization Project
```
