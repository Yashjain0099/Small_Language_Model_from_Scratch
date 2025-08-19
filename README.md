# Small Language Model (SLM) from Scratch

This project implements a **Transformer-based Small Language Model (SLM)** from scratch in **PyTorch**, inspired by GPT architectures.  
It uses the **GPT-2 tokenizer (tiktoken)** for encoding text and a custom-built transformer with causal self-attention for autoregressive text generation.

---

## 🚀 Features
- Custom **Transformer architecture** (multi-head causal self-attention, MLP, LayerNorm).
- **Tokenizer**: GPT-2 BPE tokenizer via `tiktoken`.
- **Training pipeline** with:
  - AdamW optimizer  
  - Warmup + Cosine LR scheduler  
  - Mixed precision training (AMP)  
  - Gradient accumulation & checkpointing
- **Text Generation**: autoregressive sampling with temperature & top-k filtering.
- Runs on **CPU or GPU (CUDA)** in Google Colab.

---

## 📂 Project Structure
📦 Small-Language-Model
┣ 📜 SLM_(scratch).ipynb # Colab notebook (training + generation)
┗ 📜 README.md # Documentation


---

## ⚙️ How It Works
1. **Tokenization**  
   Uses GPT-2 tokenizer (`tiktoken`) to convert raw text into tokens.  

2. **Transformer Model**  
   - Input embeddings + positional embeddings  
   - Stacked transformer blocks with self-attention  
   - Language modeling head for token prediction  

3. **Training**  
   - Trained on tokenized `.bin` dataset (train + validation splits).  
   - Optimized using AdamW with learning rate scheduling.  
   - Checkpoints and best model saved.  

4. **Generation**  
   - Autoregressive text generation from a seed prompt.  
   - Supports temperature scaling and top-k filtering.

---

## 📊 Example Output
Input Seed : A little girl went to the woods
Output : A little girl went to the woods and he was looking at the animals and he saw a little boy with a big smile on its face. He knew she would never bring medicine before.

One day, the girl called Jeff went for a walk. He saw something pretty in a nearby old structure. It was a kind of creature and he is leaning chillis and hiding. Lilly looked up at the octopus. Littleie felt very worried. She thought it looked colourful and funny. Then found a strange idea. She took the unknown and said, â€œufchieumbling, stone!â€

Ted the factory said, â€œI have a hero!â€; he was sure it will rise. 

Mary smiled and was not getting still there before. It was a powerful tw armor! And this made him even more he dream less very safe.It was a awesome day. John saw something sparkly and was a brilliant trick. It was a mighty think of something

--

## 🔧 Tech Stack
- Python
- PyTorch
- tiktoken
- Google Colab

---
