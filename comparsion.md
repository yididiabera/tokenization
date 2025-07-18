
### **1. Character Tokenizer**
**Output:** `['I', "'", 'm', ' ', 'n', 'o', 't', ' ', 'i', 'n', ' ', 'd', 'a', 'n', 'g', 'e', 'r', '.', ' ', 'I', "'", 'm', ' ', 't', 'h', 'e', ' ', 'd', 'a', 'n', 'g', 'e', 'r', '!']`  
**What it does:**  
Breaks text into individual characters, including spaces and punctuation.  
**Best for:**  
- Simple text processing  
- Languages without word boundaries (e.g., Chinese)  
**Limitation:**  
Long sequences lose word meaning (e.g., "danger" becomes 6 separate letters).

---

### **2. Whitespace Tokenizer**  
**Output:** `['I', 'am', 'not', 'in', 'danger.', 'I', 'am', 'the', 'danger!']`  
**What it does:**  
Splits text at spaces but keeps punctuation attached to words.  
**Best for:**  
- Quick prototyping  
- Basic word frequency analysis  
**Limitation:**  
Poor handling of contractions ("I'm" → "I") and punctuation ("danger!" ≠ "danger").

---

### **3. Word-Level Tokenizer**  
**Output:** `['I', "'m", 'not', 'in', 'danger', '.', 'I', "'m", 'the', 'danger', '!']`  
**What it does:**  
Splits words and handles punctuation/contractions separately.  
**Best for:**  
- Cleaner word-based analysis  
- Tasks needing punctuation awareness  
**Limitation:**  
Large vocabularies for rare words (e.g., "tokenization" = 1 unique token).

---

### **4. Basic BPE Tokenizer**  
**Output:** `["I'm n", 'o', 't', ' ', 'i', 'n', ' danger", '.', ' ', "I'm ", 't', 'h', 'e', ' danger', '!']`  
**What it does:**  
Merges frequent character pairs (e.g., "I'm" stays together but splits "not").  
**Best for:**  
- Balancing vocabulary size and meaning  
- Custom tokenization without ML models  
**Limitation:**  
Chunks can be arbitrary ("I'm n" isn’t meaningful).

---

### **5. BERT Tokenizer (WordPiece)**  
**Output:** `['i', "'", 'm', 'not', 'in', 'danger', '.', 'i', "'", 'm', 'the', 'danger', '!']`  
**What it does:**  
Splits unknown words into subwords (e.g., "playing" → "play" + "ing").  
**Best for:**  
- Transformer models like BERT  
- Handling rare words efficiently  
**Limitation:**  
Over-splits contractions ("I'm" → 3 tokens).

---

### **6. GPT-2 Tokenizer (BPE)**  
**Output:** `['i', "'m", 'Ġnot', 'Ġin', 'Ġdanger', '.', 'Ġi', "'m", 'Ġthe', 'Ġdanger', '!']`  
**What it does:**  
Uses byte-level BPE with special space symbols (`Ġ`).  
**Best for:**  
- GPT-style language models  
- Preserving capitalization/spacing  
**Limitation:**  
Space tokens (`Ġ`) can look unnatural to humans.

---
