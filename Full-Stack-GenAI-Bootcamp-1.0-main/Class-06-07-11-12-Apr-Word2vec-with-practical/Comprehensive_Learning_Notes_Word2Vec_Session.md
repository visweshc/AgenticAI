# Comprehensive Learning Notes from Word2Vec Class Session (Class 6 & 7)

## Session Overview
- **Dates**: April 11-12, 2026
- **Topics**: Word2Vec Deep Dive, CBOW vs Skip-gram, Neural Network Training, Practical Implementation
- **Instructor**: Sunny Savita

---

## 1. Quick Recap from Previous Session

### Core Concepts Revised
- **Word2Vec**: Neural network-based model for converting words to numerical vectors
- **Two Training Methods**: 
  1. **CBOW** (Continuous Bag of Words)
  2. **Skip-gram**
- **Vocabulary**: Set of unique words from corpus (case-sensitive)
- **Window Size**: Hyperparameter defining context words around target word
- **One-Hot Encoding**: Binary representation of words based on vocabulary

### Example Sentence
```
"follow Sunny for the generative AI"
```

**Vocabulary**: {follow, Sunny, for, the, generative, AI}

---

## 2. CBOW (Continuous Bag of Words) - Detailed Architecture

### Definition
- **Architecture Type**: Many-to-One
- **Input**: Multiple context words
- **Output**: Single target word

### Data Preparation (Window Size = 1)

| Context (Input) | Target (Output) |
|----------------|-----------------|
| [follow, 4] | Sunny |
| [Sunny, for] | 4 |
| [4, the] | for |
| [for, generative] | the |
| [the, AI] | generative |

### Neural Network Structure

```
Input Layer (One-hot encoded context words)
    |
    | Matrix Multiplication
    v
Hidden Layer (Dense layer without activation)
    |
    | Output = Embedding
    v
Output Layer (Softmax for probability)
```

### Key Components

**Input Layer**
- Can accept multiple words in parallel
- Number of words depends on window size
- Each word is one-hot encoded

**Hidden Layer**
- **Only ONE hidden layer** (as per research paper)
- Number of neurons = embedding dimension (hyperparameter)
- Examples: 3, 10, 30, 100, 150, 300, 500
- Google's model used 300 dimensions
- **No activation function** in hidden layer (as per paper)

**Output Layer**
- Uses **Softmax activation** for probability distribution
- Outputs probability for each word in vocabulary

### Weight Matrices

**Input to Hidden Weights**
- Size: Vocabulary Size × Hidden Neurons
- Example: 5 × 3 (5 words in vocabulary, 3 hidden neurons)
- Total weights: 15

**Hidden to Output Weights**
- Size: Hidden Neurons × Vocabulary Size
- Example: 3 × 5

### Training Process

**Forward Propagation**
1. Input words converted to one-hot encoded vectors
2. Multiply with input-to-hidden weight matrix
3. Hidden layer output = embedding (intermediate representation)
4. Multiply with hidden-to-output weight matrix
5. Apply softmax to get predicted probabilities

**Backward Propagation**
1. Calculate loss: Predicted - Actual
2. Use optimizer (Gradient Descent) to minimize loss
3. Update weights to reduce loss
4. Goal: Find optimal weights with minimum loss

**Loss Function**
```
Loss = Predicted - Actual
Goal: Minimize Loss
```

**Global Minima**
- Point where loss is minimum
- Weights are optimized at this point
- Represents best possible embeddings

### Key Insight
> **"The output of the hidden layer itself becomes the word embedding"**

This intermediate representation is the meaningful vector that captures semantic relationships.

---

## 3. Skip-gram - Detailed Architecture

### Definition
- **Architecture Type**: One-to-Many
- **Input**: Single target word
- **Output**: Multiple context words

### Data Preparation (Window Size = 1)

| Context (Input) | Target (Output) |
|----------------|-----------------|
| Sunny | [follow, 4] |
| 4 | [Sunny, for] |
| for | [4, the] |
| the | [for, generative] |
| generative | [the, AI] |

### Neural Network Structure

```
Input Layer (One-hot encoded target word)
    |
    | Matrix Multiplication
    v
Hidden Layer (Dense layer without activation)
    |
    | Output = Embedding
    v
Output Layer (Multiple Softmax outputs, one per context word)
```

### Key Differences from CBOW

**Input/Output Relationship**
- **CBOW**: Multiple context words predict one target
- **Skip-gram**: One target word predicts multiple context words

**Learning Quality**
- Skip-gram provides **better learning** because:
  - One word maps to multiple targets
  - More training examples per word
  - Better captures relationships
  - Proven to give better accuracy on large datasets

**Research Paper Finding**
> "When they took the large dataset, with Skip-gram they were getting a good result, the good accuracy."

### Training Process (Same as CBOW)
1. Forward propagation
2. Loss calculation
3. Backward propagation
4. Weight optimization
5. Minimize loss

### Example with Window Size = 2

**Sentence**: "My name is Sunny and AI is my favorite subject"

**CBOW Data**:
- Context: [My, name, and, AI] (4 words) | Target: is
- Context: [name, is, AI, my] (4 words) | Target: Sunny

**Skip-gram Data**:
- Input: Sunny | Output: [name, is, and, AI] (4 words)
- Input: is | Output: [Sunny, and, AI, my] (4 words)

### Comparison: Which Provides Better Learning?

**Answer**: Skip-gram

**Why?**
- More context per word
- Better handling of rare words
- Stronger embeddings
- More training examples

---

## 4. Neural Network Training - Complete Workflow

### Step-by-Step Process

**Step 1: Initialize Weights**
- Weights initialized randomly
- Random initialization is standard practice

**Step 2: Forward Propagation**
```
Input (One-hot) × Weight Matrix (Input to Hidden) = Hidden Output
Hidden Output × Weight Matrix (Hidden to Output) = Predicted Output
```

**Step 3: Loss Calculation**
```
Loss = Predicted Output - Actual Output
```

**Step 4: Backward Propagation**
- Calculate gradients
- Update weights using optimizer
- Common optimizers: Gradient Descent, Adam, RMSprop

**Step 5: Iterate (Epochs)**
- Epoch = Forward + Backward propagation
- Multiple iterations to optimize weights

**Step 6: Freeze Model**
- After training, model is frozen (weights fixed)
- Ready for inference/prediction

### Gradient Descent Visualization

```
Loss
  ^
  |      /\
  |     /  \
  |    /    \
  |   /      \
  |  /        \______ Global Minima
  | /
  +------------------> Weights
```

- **Goal**: Reach global minima
- **At global minima**: Loss is minimum, weights are optimal

---

## 5. Embedding vs Encoding - Clear Distinction

### Encoding (Static Methods)
- **Examples**: One-hot encoding, Bag of Words, TF-IDF
- **Nature**: Manual, counting-based
- **No training involved**
- **No neural network**
- **Static representations**

### Embedding (Learned Methods)
- **Examples**: Word2Vec, GloVe, BERT embeddings
- **Nature**: Dynamic, learned through training
- **Neural network-based**
- **Captures semantic meaning**
- **Context-aware**

### Key Difference
> **Encoding = Manual counting (no learning)**
> **Embedding = Neural network learning (captures meaning)**

---

## 6. Practical Implementation with Gensim

### Required Libraries

```python
import pandas as pd
import numpy as np
import os
import gensim
import nltk
```

### NLTK Setup

```python
# Download all NLTK resources
nltk.download('all')

# Import required functions
from nltk.tokenize import sent_tokenize
from gensim.utils import simple_preprocess
```

### Data Loading

```python
# List files in data folder
files = os.listdir('data')
print(files)  # Output: ['story1.txt', 'story2.txt']

# Read a file
with open('data/story1.txt', 'r', encoding='unicode_escape') as f:
    corpus = f.read()
```

### Text Preprocessing

```python
# Tokenize sentences
sentences = sent_tokenize(corpus)

# Preprocess each sentence (tokenize words, lowercase, remove punctuation)
story = []
for sent in sentences:
    story.append(simple_preprocess(sent))
```

### Custom Word2Vec Model Training

**CBOW (Default)**:
```python
from gensim.models import Word2Vec

custom_model = Word2Vec(
    window=10,        # Context window size
    min_count=1,      # Minimum word frequency
    vector_size=150  # Embedding dimensions
)

# Build vocabulary
custom_model.build_vocab(story)

# Train model
custom_model.train(story, 
                  total_examples=custom_model.corpus_count, 
                  epochs=10)
```

**Skip-gram**:
```python
custom_model = Word2Vec(
    window=10,
    min_count=1,
    vector_size=150,
    sg=1  # Set to 1 for Skip-gram
)
```

### Accessing Embeddings

```python
# Get vector for a word
vector = custom_model.wv['sunny']
print(vector.shape)  # Output: (150,)

# Find similar words
similar_words = custom_model.wv.most_similar('sunny')
print(similar_words)
```

### Model Parameters Explained

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `window` | Context window size | 1, 2, 5, 10, 20 |
| `min_count` | Minimum word frequency | 1, 5, 10 |
| `vector_size` | Embedding dimensions | 100, 150, 300 |
| `sg` | Training method | 0 (CBOW), 1 (Skip-gram) |
| `epochs` | Training iterations | 10, 20, 50 |

---

## 7. Hyperparameters - Complete Guide

### What are Hyperparameters?
- Parameters set BEFORE training
- Not learned during training
- Experimental values
- Need tuning for optimal performance

### Key Hyperparameters in Word2Vec

**1. Window Size**
- **Definition**: Number of context words to consider
- **Impact**: Larger window = more context but slower training
- **Typical Values**: 5-30
- **How to Choose**: Experiment and evaluate accuracy

**2. Vector Size (Embedding Dimension)**
- **Definition**: Number of neurons in hidden layer
- **Impact**: Larger = more features but more computation
- **Typical Values**: 100-300 (Google used 300)
- **How to Choose**: 
  - Start with 10-50 for small datasets
  - Use 100-300 for larger datasets
  - Plot accuracy graph to find optimal

**3. min_count**
- **Definition**: Minimum frequency for word to be included
- **Impact**: Filters out rare words
- **Typical Values**: 1-10
- **How to Choose**: Depends on corpus size

**4. sg (Training Method)**
- **Definition**: 0 for CBOW, 1 for Skip-gram
- **Impact**: Skip-gram better for rare words
- **How to Choose**: Use Skip-gram for better accuracy

**5. epochs**
- **Definition**: Number of training iterations
- **Impact**: More epochs = better learning (to a point)
- **Typical Values**: 10-100
- **How to Choose**: Monitor loss, stop when it plateaus

### Hyperparameter Tuning Methods

**Grid Search**
- Try all combinations of parameter values
- Computationally expensive
- Guaranteed to find best combination

**Random Search**
- Try random combinations
- Faster than grid search
- Often finds good solutions

**Manual Tuning**
- Start with reasonable defaults
- Adjust one parameter at a time
- Monitor performance metrics

---

## 8. Important Questions & Answers

### Q1: Why do we initialize weights randomly?
**A**: Random initialization is standard practice in neural networks. It breaks symmetry and allows different neurons to learn different features.

### Q2: How do we decide weight matrix sizes (e.g., 5×3, 3×5)?
**A**: 
- 5×3: 5 input words (vocabulary size) × 3 hidden neurons
- 3×5: 3 hidden neurons × 5 output words (vocabulary size)
- In fully connected network: Total weights = inputs × neurons

### Q3: Why no activation function in hidden layer?
**A**: As per the original Word2Vec research paper, they didn't use activation function in the embedding layer. The linear transformation preserves the learned relationships.

### Q4: What is an "epoch"?
**A**: One complete pass through the entire training dataset. Epoch = Forward propagation + Backward propagation.

### Q5: What happens after training?
**A**: 
- Model is "frozen" (weights fixed)
- Ready for inference
- Can generate embeddings for new words
- Weights represent learned semantic relationships

### Q6: How does Skip-gram handle same word appearing multiple times?
**A**: The word appears multiple times in training data with different contexts. The model averages all contexts to create the final embedding.

### Q7: Which is better: CBOW or Skip-gram?
**A**: 
- **Skip-gram**: Better for rare words, higher accuracy on large datasets
- **CBOW**: Faster training, good for frequent words
- **Recommendation**: Use Skip-gram for better embeddings

### Q8: Why is embedding called "intermediate representation"?
**A**: Because it's the output of the hidden layer, which is intermediate between input and output layers. It captures learned features before final prediction.

---

## 9. Key Takeaways for Interviews

### Must-Know Concepts

1. **Word2Vec Definition**
   - Neural network-based word embedding technique
   - Converts words to dense vectors
   - Captures semantic relationships

2. **CBOW vs Skip-gram**
   - CBOW: Many-to-one (context predicts target)
   - Skip-gram: One-to-many (target predicts context)
   - Skip-gram provides better learning

3. **Neural Network Architecture**
   - Input layer (one-hot encoded)
   - Hidden layer (no activation, produces embeddings)
   - Output layer (softmax for probability)

4. **Training Process**
   - Forward propagation (predict)
   - Loss calculation (predicted - actual)
   - Backward propagation (update weights)
   - Goal: Minimize loss, find optimal weights

5. **Hyperparameters**
   - Window size, vector size, min_count, sg, epochs
   - Need experimental tuning
   - No fixed "best" values

6. **Embedding vs Encoding**
   - Encoding: Static, manual (one-hot, TF-IDF)
   - Embedding: Learned, neural network-based

### Code Snippets for Interviews

```python
# CBOW
model = Word2Vec(window=10, min_count=1, vector_size=150, sg=0)

# Skip-gram
model = Word2Vec(window=10, min_count=1, vector_size=150, sg=1)

# Get embedding
vector = model.wv['word']

# Find similar words
similar = model.wv.most_similar('word')
```

---

## 10. Practical Tips & Best Practices

### Data Preparation
1. Always lowercase text (unless case matters)
2. Remove punctuation and special characters
3. Tokenize sentences first, then words
4. Use appropriate encoding when reading files

### Model Training
1. Start with small window size (5-10)
2. Use 100-300 dimensions for embeddings
3. Train for 10-50 epochs
4. Monitor loss during training
5. Use Skip-gram for better results

### Common Issues & Solutions

**Issue**: Empty corpus when reading file
**Solution**: File cursor moves after first read, need to re-open file

**Issue**: NLTK resources not found
**Solution**: Run `nltk.download('all')` first

**Issue**: Poor similarity results
**Solution**: Increase window size, train more epochs, use larger dataset

---

## 11. Future Topics Roadmap

### Upcoming Topics
1. **Transformer-based models** (BERT, GPT)
2. **Similarity search** (keyword vs semantic)
3. **Transformer architecture** (3-4 classes)
4. **LLM architectures** (Llama, multimodal)
5. **Fine-tuning** (Hugging Face, vLLM)
6. **RAG** (Retrieval Augmented Generation)
7. **Agents and MCP**
8. **LLM Ops**

### Timeline
- 8-10 classes for transformer and LLM basics
- 1-1.5 months total for foundational topics
- Then move to practical applications

---

## 12. Important Notes

### Python Version
- Use Python 3.11 for compatibility
- Some libraries (like Gensim) may not support 3.12

### Resources
- GitHub repository for all assignments and code
- Course dashboard for recordings and materials
- Notion notes converted to PDF in GitHub
- Research paper on Word2Vec available

### Assignments
- Weekly assignments with flexible timeline
- Solutions provided weekly
- Monthly mega challenge with prizes planned
- Focus on learning through practical implementation

### Learning Approach
- This topic requires **revision**
- Core mathematical understanding takes time
- Even experienced practitioners learn something new each time
- Don't panic if not fully understood in first pass
- Use prerequisite materials for neural network basics

---

## 13. Summary of Key Equations

### Matrix Multiplication
```
Hidden Output = Input Vector × Weight Matrix (Input to Hidden)
Predicted Output = Hidden Output × Weight Matrix (Hidden to Output)
```

### Loss Calculation
```
Loss = Predicted Output - Actual Output
```

### Embedding Generation
```
Embedding = Input Vector × Trained Weight Matrix (Input to Hidden)
```

---

## 14. Interview Questions Bank

### Basic Questions
1. What is Word2Vec?
2. What are the two types of Word2Vec?
3. What is the difference between CBOW and Skip-gram?
4. What is window size?
5. What is one-hot encoding?

### Intermediate Questions
6. How does Word2Vec training work?
7. What is the neural network architecture of Word2Vec?
8. Why is there no activation function in the hidden layer?
9. What are hyperparameters in Word2Vec?
10. How do you choose hyperparameters?

### Advanced Questions
11. Explain forward and backward propagation in Word2Vec
12. What is global minima in the context of Word2Vec?
13. How are embeddings different from encodings?
14. Why does Skip-gram provide better learning than CBOW?
15. How do you handle rare words in Word2Vec?

### Practical Questions
16. Write code to train a Word2Vec model
17. How do you get embeddings for a word?
18. How do you find similar words?
19. How do you prepare data for Word2Vec?
20. What preprocessing steps are needed?

---

## 15. Quick Reference

### CBOW
```python
model = Word2Vec(window=10, min_count=1, vector_size=150, sg=0)
```

### Skip-gram
```python
model = Word2Vec(window=10, min_count=1, vector_size=150, sg=1)
```

### Get Embedding
```python
vector = model.wv['word']
```

### Find Similar Words
```python
similar = model.wv.most_similar('word')
```

---

## Conclusion

This session provided a comprehensive deep dive into Word2Vec, covering both theoretical concepts and practical implementation. Key takeaways:

1. **Word2Vec** uses neural networks to learn word representations
2. **Skip-gram** generally provides better embeddings than CBOW
3. **Training** involves forward/backward propagation to minimize loss
4. **Hyperparameters** need experimental tuning
5. **Practical implementation** is straightforward with Gensim

Remember: This topic requires revision. Don't worry if you don't understand everything in one pass. Practice with code and revisit the concepts multiple times for deep understanding.

---

**Session Date**: April 11-12, 2026
**Instructor**: Sunny Savita
**Next Topic**: Transformer Architecture
