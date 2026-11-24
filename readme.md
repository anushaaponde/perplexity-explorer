# ⚡ Perplexity Explorer PRO

> An interactive, visual, and beginner-friendly deep dive into how modern language models compute token probability, surprisal, and perplexity.

![Perplexity Explorer Banner](https://img.shields.io/badge/Model-Xenova%2Fdistilgpt2-indigo?style=for-the-badge&logo=huggingface)
![Transformers.js](https://img.shields.io/badge/Runtime-Transformers.js%20v3%20ONNX-blue?style=for-the-badge)
![Zero Dependencies](https://img.shields.io/badge/Build-Single--File%20(index.html)-emerald?style=for-the-badge)
![Theme](https://img.shields.io/badge/Theme-Dark%20%2F%20Light-purple?style=for-the-badge)

---

## 🌟 Overview

**Perplexity Explorer PRO** is an educational, production-grade web application built to demystify one of the most fundamental yet misunderstood evaluation metrics in Natural Language Processing (NLP) and Large Language Models (LLMs): **Perplexity**.

Everything runs **100% locally in your browser** using [Transformers.js v3](https://huggingface.co/docs/transformers.js) and an ONNX 8-bit quantized version of **DistilGPT-2**. There are **no external servers, no tracking, and no backend API keys required**.

---

## 🚀 Key Improvements Over the Original App

The original version provided bare-bones cards with a plain light theme and a static table. **Perplexity Explorer PRO** takes this experience to another level:

| Feature / Dimension | Original Explorer | Perplexity Explorer PRO |
| :--- | :--- | :--- |
| **Design System** | Generic Bootstrap-style flat light theme | **Glassmorphic futuristic UI** with full **Dark Mode** & **Light Mode** toggle, persisted in `localStorage` |
| **Visual Intuition** | Raw numbers in a plain table | **Dynamic Surprisal Heatmap Ribbon**: Tokens highlighted from Emerald (high confidence) to Crimson (high surprise) |
| **Instant Evaluation** | Forced ~240MB model download on click | **Instant Demo Preview mode** for fast learning on slow networks + real ONNX DistilGPT-2 loader |
| **Next-Token Generation** | Static top 8 predictions list | **Click-to-Append Interactive Autocomplete**: Click candidate tokens to generate text token-by-token and watch perplexity evolve |
| **Softmax Temperature** | Fixed temperature = 1.0 | **Interactive Temperature Slider (0.1× to 2.0×)** with live softmax probability rescaling |
| **Hand-Crafted Sandbox** | Manual typing in plain text boxes | **Dual Slider & Numeric Controls**, real-time validation, **Normalize** & **Randomize** utility buttons |
| **Audio Feedback** | Silent | Toggleable **Web Audio API sound feedback** (subtle clicks, pops, and chimes) |
| **Export & Sharing** | None | **One-click "Copy Report" button** that outputs structured Markdown tables and metrics |
| **Educational Theory** | Minimal single-sentence notes | **Full Math Academy section**: The 20-Questions analogy, Geometric vs. Arithmetic mean, and Real-World LLM Scale |
| **Keyboard Accessibility** | Mouse-only | <kbd>Ctrl</kbd>/<kbd>Cmd</kbd> + <kbd>Enter</kbd> to evaluate, keyboard navigable controls |

---

## 📐 The Mathematics of Perplexity

### 1. Surprisal (Information Content)
Given a sequence of tokens $W = (w_1, w_2, \dots, w_N)$, the model calculates the probability of each token conditioned on all preceding tokens:

$$P(w_k \mid w_1, w_2, \dots, w_{k-1})$$

The **surprisal** (also called self-information) of token $w_k$ is the reciprocal of its probability:

$$\text{Surprisal}(w_k) = \frac{1}{P(w_k \mid w_{<k})}$$

- If a token is guaranteed ($P = 1.0$), surprisal is $1.0$ (no surprise).
- If a token is improbable ($P = 0.001$), surprisal is $1000$ (massive shock).

### 2. Sequence Probability & Length Normalization
The joint probability of the entire sequence is the product of all conditioned token probabilities:

$$P(W) = \prod_{k=1}^N P(w_k \mid w_{<k})$$

Taking the reciprocal product gives the total surprisal:

$$\prod_{k=1}^N \frac{1}{P(w_k \mid w_{<k})}$$

To make this metric comparable between sentences of different lengths, we take the **geometric mean** by raising the product to the power of $\frac{1}{N}$:

$$\text{Perplexity}(W) = \left( \prod_{k=1}^N \frac{1}{P(w_k \mid w_{<k})} \right)^{\frac{1}{N}}$$

### 3. Log-Domain Numerical Stability
In practice, multiplying dozens of probabilities smaller than $0.1$ causes numerical underflow (rounding down to zero). We compute perplexity in the logarithmic domain:

$$\ln(\text{Perplexity}) = \frac{1}{N} \sum_{k=1}^N -\ln P(w_k \mid w_{<k})$$

$$\text{Perplexity} = \exp\left( \frac{1}{N} \sum_{k=1}^N -\ln P(w_k \mid w_{<k}) \right)$$

This is identical to $\exp(\text{Cross-Entropy Loss})$.

---

## 🧩 Architectural Breakdown of the 5 Sections

### Section 1: Prompt & Language Model Studio
- **Input Textarea**: Real-time character and word count tracking.
- **Curated Linguistic Presets**:
  - *Fact*: "The capital of France is Paris" (High probability, low perplexity)
  - *Natural*: "I love Python programming" (Fluent syntax)
  - *Scrambled*: "Python love I" (Syntactic anomaly, higher perplexity)
  - *Chaotic*: "Banana elephant quantum toaster galaxy" (Semantic nonsense, extreme perplexity)
  - *Repetitive*: "buffalo buffalo buffalo buffalo buffalo"
- **Model Download Engine**: Streams ONNX weights with progress bars and caches models in the browser's IndexedDB.

### Section 2: Surprisal Heatmap & Step-by-Step Breakdown
- **Surprisal Heatmap Ribbon**: Every token rendered as a colored badge with surprisal factors.
- **Summary Metrics Grid**: Perplexity (PPL), Total Tokens ($N$), Average Cross-Entropy, and Most Surprising Token.
- **Interactive Data Table**: Shows Token, Token ID, Softmax Probability with mini visual progress bar, Surprisal ($1/P$), and Information in bits ($-\log_2 P$).
- **3-Step Math Pipeline**: Transparent breakdown showing the exact multiplication, length normalization, and final geometric mean.

### Section 3: Next-Token Predictive Horizon
- Predicts what the model expects to come next based on the final token's logits.
- **Interactive Autocomplete**: Click the **"+ Append"** button on any candidate token to immediately append it to your sentence and re-evaluate!
- **Temperature Slider**: Adjust softmax sharpness from $0.1$ (greedy / peaked) to $2.0$ (uniform / flattened).

### Section 4: Interactive Probability Sandbox
- Design custom probability distributions by hand.
- Sliders and numeric inputs update bidirectionally in real time.
- Presets:
  - *"I love Python"* ($0.5, 0.8, 0.2 \to \text{PPL } \approx 2.32$)
  - *8-Sided Fair Die* ($8 \times 0.125 \to \text{PPL } = 8.00$)
  - *Black Swan Shock* ($0.95, 0.95, 0.001 \to$ reveals how one low-probability word dominates perplexity)
  - *Coin Flips* ($0.5, 0.5, 0.5 \to \text{PPL } = 2.00$)
- Utilities: **Normalize** (scales probabilities to sum to 1.0) and **Randomize**.

### Section 5: The Perplexity Intuition Guide
- The **20-Questions & Fair Dice** mental model.
- Why geometric mean is used instead of arithmetic mean.
- Real-world language model benchmark scale (from 1.0 Oracle to 50,257 full vocabulary uniform noise).
- Frequently Asked Questions accordion.

---

## 💻 How to Run

Because this project is strictly self-contained within **one file (`index.html`)**, running it requires zero package managers, compilers, or build steps:

### Option 1: Direct File Open
Simply double-click `index.html` or open it in any modern browser:
```bash
open index.html # on macOS
# or
xdg-open index.html # on Linux
```

### Option 2: Local HTTP Server (Recommended for ONNX caching)
Running through a local web server allows the browser to utilize `IndexedDB` caching for the ONNX model:
```bash
python3 -m http.server 8000
```
Then navigate to:
```
http://localhost:8000
```

---

## 🛠️ Technology Stack

- **Markup & Styling**: Semantic HTML5, CSS custom variables (theming), CSS Grid & Flexbox.
- **Typography**: Google Fonts (`Plus Jakarta Sans`, `Inter`, `JetBrains Mono`).
- **Icons**: Inline SVGs (lightweight, zero network overhead).
- **Inference Engine**: Hugging Face `@huggingface/transformers@3` CDN.
- **Model**: `Xenova/distilgpt2` (8-bit Quantized ONNX format, ~240MB).
- **Sound**: Native HTML5 Web Audio API synthesizer.

---

## 📄 License

MIT License. Free to use, modify, and distribute for educational, research, and commercial purposes.
