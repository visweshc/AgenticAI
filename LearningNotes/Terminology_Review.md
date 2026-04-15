# Terminology Review: Text Encoding, Tokenization, and Embeddings

## Topic: Modern Route-Full Stack Gen Ai And Agentic AI Bootcamp

### Tokenization

Tokenization is the foundational step in Natural Language Processing (NLP). It involves breaking down a stream of text into smaller units called **tokens**. These tokens can be words, characters, or sub-words (parts of words). For a machine learning model to process and understand human language, the text must first be converted into a numerical format, and tokenization is the very first step in this process.

There are several common strategies for tokenization:

*   **Word Tokenization**: This method splits the text based on whitespace and punctuation. While simple and intuitive, it can struggle with complex words, hyphenated terms, or languages that don't use spaces to separate words.
*   **Character Tokenization**: This method breaks the text down into individual characters. This approach avoids the problem of "unknown" words, but it results in very long sequences, which can be computationally expensive for models to process.
*   **Subword Tokenization**: This is a hybrid approach that balances the pros and cons of word and character tokenization. It keeps common words as single tokens while breaking down rare or complex words into smaller, meaningful sub-units. This is the most prevalent method used in modern NLP models like Transformers. Popular algorithms for subword tokenization include:
    *   **Byte-Pair Encoding (BPE)**
    *   **WordPiece** (used by BERT)
    *   **SentencePiece**

### Text Encoding

Once the text has been tokenized, the next step is **text encoding**. This is the process of converting the string tokens into a sequence of numerical IDs. A **vocabulary** is built from the training corpus, which maps each unique token to a specific integer.

For example, given a small vocabulary:
`{"<start>": 0, "hello": 1, "world": 2, "<end>": 3, "<unk>": 4}`

The sentence "hello world" would be encoded into the integer sequence: `[1, 2]`.

Models often use special tokens to handle various situations:

*   **`<pad>` (Padding)**: Since models often process text in batches, all input sequences in a batch must have the same length. Shorter sequences are "padded" with this token to match the length of the longest sequence.
*   **`<unk>` (Unknown)**: If a token appears in the input text but is not present in the model's vocabulary, it is mapped to the `<unk>` token's ID.
*   **`<sos>` / `<bos>` (Start of Sequence/Beginning of Sentence)**: A special token that marks the beginning of a text sequence.
*   **`<eos>` (End of Sequence)**: A special token that marks the end of a text sequence.

This sequence of integer IDs is the final input that is fed into the model's first layer.

### Embeddings

While text encoding provides a numerical representation, the integer IDs are arbitrary and do not capture any semantic meaning. For instance, the words "cat" and "dog" are semantically related, but their assigned integer IDs could be completely random and far apart (e.g., 5 and 987). This is the problem that **embeddings** solve.

**Embeddings are dense vector representations of tokens.** Each token ID from the vocabulary is mapped to a high-dimensional vector (a list of floating-point numbers). This vector represents the token's learned meaning in a continuous vector space.

Key characteristics of embeddings:

*   **Semantic Relationships**: The power of embeddings lies in their ability to capture the semantic relationships between words. In the vector space, words with similar meanings will be located close to each other. This allows for algebraic operations on word meanings. The classic example is `vector('king') - vector('man') + vector('woman')`, which results in a vector very close to `vector('queen')`.
*   **Learned from Data**: These vectors are not manually created. They are learned by the model during the training process. The model adjusts the values in the embedding vectors to minimize its prediction error on the training data. This process allows the model to learn rich, contextual representations of words based on how they are used in the text.
*   **Dimensionality**: The size of the embedding vector (its dimensionality) is a hyperparameter. Common dimensions range from a few hundred to several thousand (e.g., 768 for BERT, 1536 for OpenAI's `text-embedding-ada-002`).

In a neural network, the first layer is typically an **Embedding Layer**. This layer acts as a lookup table, taking the integer-encoded tokens as input and outputting the corresponding dense embedding vectors. These vectors, which now contain semantic information, are then passed through the subsequent layers of the network for further processing.

---

## Key Distinctions: Encoding vs. Embedding

### Encoding
**Definition**: Numerical representation of data

**Encoding** is the process of converting raw data into numerical form (often sparse) without necessarily preserving semantic meaning.

**Characteristics**:
- Converts raw data into numerical form
- Often results in sparse representations
- May not capture semantic relationships
- Used as input to models

**Methods**: Count-based, matrix factorization-based

### Embedding
**Definition**: Meaningful numerical representation

**Embedding** is the process of creating dense vector representations that capture semantic meaning and relationships.

**Characteristics**:
- Creates dense vector representations
- Captures semantic meaning
- Preserves relationships between entities
- Learned through neural networks or transformers

**Usage**: Vector → Neural network or Transformer

---

## Encoding Methods

### 1. One-Hot Encoding ✓
**Description**: Represents each word as a binary vector where only one position is "hot" (1) and all others are "cold" (0).

**Example**:
- Vocabulary: ["cat", "dog", "bird"]
- "cat" → [1, 0, 0]
- "dog" → [0, 1, 0]
- "bird" → [0, 0, 1]

**Limitations**:
- Very sparse representation
- No semantic meaning captured
- Vector size equals vocabulary size
- Cannot handle unknown words

### 2. Bag of Words (BOW) ✓
**Description**: Represents text as a collection of words, counting their frequency while ignoring grammar and word order.

**Example**:
- Sentence: "the cat sat on the mat"
- BOW: {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}

**Characteristics**:
- Count-based method
- Ignores word order
- Simple and fast
- Good for document classification

### 3. TF-IDF (Term Frequency-Inverse Document Frequency) ✓
**Description**: Weighs words based on their importance in a document relative to a corpus.

**Formula**:
- TF-IDF = TF × IDF
- TF (Term Frequency) = (Number of times term appears in document) / (Total terms in document)
- IDF (Inverse Document Frequency) = log(Total documents / Documents containing term)

**Use Case**: Helps identify important words in a document

### 4. N-grams
**Description**: Sequences of N consecutive words or characters.

**Examples**:
- Unigrams (1-gram): ["the", "cat", "sat"]
- Bigrams (2-gram): ["the cat", "cat sat"]
- Trigrams (3-gram): ["the cat sat"]

**Benefit**: Captures some word order and context

### 5. BM25
**Description**: An improved version of TF-IDF, technically a ranking algorithm rather than just an encoding method.

**Characteristics**:
- Ranking algorithm
- Better than basic TF-IDF
- Used in search engines
- Considers document length normalization

---

## Embedding Methods

### 1. GloVe (Global Vectors)
**Description**: GloVe is a count-based + matrix factorization-based method for creating word embeddings.

**Characteristics**:
- Combines global matrix factorization and local context window methods
- Pre-trained on large corpora (Wikipedia, Common Crawl)
- Captures word co-occurrence statistics
- Creates dense vector representations

**Key Feature**: Uses global word-word co-occurrence statistics

### 2. Word2Vec
**Description**: Neural network-based method for learning word embeddings.

**Two Architectures**:
- **CBOW (Continuous Bag of Words)**: Predicts target word from context words
- **Skip-gram**: Predicts context words from target word

**Characteristics**:
- Creates dense vectors
- Captures semantic relationships
- Fast training
- Good for word analogies

### 3. FastText
**Description**: Extension of Word2Vec that represents words as bags of character n-grams.

**Advantages**:
- Handles out-of-vocabulary words
- Works well with rare words
- Captures morphological information

### 4. Contextual Embeddings (Modern Approach)
**Examples**: BERT, GPT, ELMo

**Key Difference**: Creates different embeddings for the same word based on context.

**Example**:
- "bank" in "river bank" vs. "bank account" gets different embeddings

---

## Practical Applications

### Count-Based Methods
- **Use Cases**: Document classification, spam detection, simple text analysis
- **Pros**: Fast, interpretable, low computational cost
- **Cons**: Sparse, no semantic meaning, large dimensionality

### Neural Network / Transformer-Based Methods
- **Use Cases**: Machine translation, sentiment analysis, question answering, text generation
- **Pros**: Dense representations, captures semantics, handles context
- **Cons**: Requires more computational resources, needs training data

---

## Key Terminology Glossary

| Term | Definition |
|------|------------|
| **Token** | A single unit of text (word, subword, or character) |
| **Vocabulary** | The set of all unique tokens known to the model |
| **Sparse Representation** | A vector with mostly zero values (e.g., one-hot encoding) |
| **Dense Representation** | A vector with mostly non-zero values (e.g., embeddings) |
| **Semantic Meaning** | The actual meaning or concept represented by a word |
| **Vector Space** | A mathematical space where embeddings exist |
| **Dimensionality** | The size/length of an embedding vector |
| **Corpus** | A large collection of text documents |
| **Context Window** | The surrounding words considered when learning embeddings |
| **Frequency** | How often a word appears in text |
| **Matrix Factorization** | Decomposing a matrix into smaller matrices |
| **Ranking Algorithm** | A method to order items by relevance |

---

## Summary: The Pipeline

1. **Raw Text** → "Hello world"
2. **Tokenization** → ["Hello", "world"]
3. **Text Encoding** → [1, 2] (integer IDs)
4. **Embedding** → [[0.2, 0.5, 0.1, ...], [0.3, 0.1, 0.8, ...]] (dense vectors)
5. **Model Processing** → Neural Network or Transformer
6. **Output** → Predictions, classifications, or generated text

---

## Temperature

### Definition
**Temperature** is a hyperparameter that controls the randomness and creativity of a model's output by adjusting the probability distribution over possible next tokens during text generation.

### How It Works

Temperature is applied to the logits (raw prediction scores) before converting them to probabilities using the softmax function:

**Formula**: `probability = softmax(logits / temperature)`

- **Lower Temperature (0.1 - 0.7)**: Makes the distribution "sharper" - the model becomes more confident and deterministic
- **Higher Temperature (0.8 - 2.0)**: Makes the distribution "flatter" - the model becomes more random and creative
- **Temperature = 1.0**: No modification to the original probability distribution

### Temperature Values and Their Effects

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| **0.0** | Deterministic (always picks highest probability) | Factual answers, code generation |
| **0.1 - 0.3** | Very focused, minimal randomness | Math problems, translations |
| **0.5 - 0.7** | Balanced, slightly creative | General Q&A, summaries |
| **0.8 - 1.0** | More creative, diverse outputs | Creative writing, brainstorming |
| **1.2 - 2.0** | Very random, experimental | Poetry, experimental content |

### Examples

**Input**: "The capital of France is"

- **Temperature 0.1**: "Paris." (deterministic, factual)
- **Temperature 0.7**: "Paris, a beautiful city known for its culture."
- **Temperature 1.5**: "Paris, though some might argue Lyon has its charms too!"

### Practical Applications

**Low Temperature (0.0 - 0.5)**:
- Code generation
- Mathematical calculations
- Factual question answering
- Data extraction
- Translation tasks

**Medium Temperature (0.6 - 0.9)**:
- General chatbots
- Content summarization
- Email writing
- Customer support

**High Temperature (1.0 - 2.0)**:
- Creative writing
- Story generation
- Brainstorming ideas
- Poetry and artistic content

### Temperature in Different Contexts

**1. Text Generation (LLMs)**:
Controls creativity vs. accuracy in generated text

**2. Model Distillation**:
Used to "soften" probability distributions so the student model can learn more nuanced information from the teacher

**3. Sampling Strategies**:
Often combined with other techniques:
- **Top-k sampling**: Sample from top k most likely tokens
- **Top-p (nucleus) sampling**: Sample from smallest set of tokens whose cumulative probability exceeds p

### Key Points

- Temperature does NOT change what the model knows
- It only affects how the model selects from what it already knows
- Temperature = 0 is greedy decoding (always pick the most likely token)
- Higher temperature increases diversity but may reduce coherence
- Different tasks require different temperature settings

### Related Parameters

- **Top-p (nucleus sampling)**: Limits sampling to tokens with cumulative probability p
- **Top-k**: Limits sampling to the k most likely tokens
- **Frequency penalty**: Reduces likelihood of repeating tokens
- **Presence penalty**: Encourages introducing new topics

---

## Retrieval-Augmented Generation (RAG)

### Key Concepts of RAG
**Architecture**: A RAG system connects a Large Language Model (LLM) to a vector database. When a query is made, the LLM retrieves relevant context from this database to generate a more informed and accurate output.

**Core Components**: The RAG pipeline consists of three main stages:
1. Document Ingestion
2. Query Processing
3. Generation

### Document Ingestion and Pre-processing
This is the first and foundational phase, which involves preparing and storing data in the vector database.

- **Data Sources**: Information is gathered from various sources such as PDFs, CSV files, websites, and internal company documents.
- **Document Splitting (Chunking)**:
  - The source documents are broken down into smaller, manageable "chunks."
  - This is necessary because LLMs have a limited context window (the amount of text they can process at once). Providing smaller, relevant chunks leads to better performance.
- **Embedding Generation**:
  - Each text chunk is converted into a numerical vector using an embedding model (e.g., from OpenAI or Hugging Face). These vectors represent the semantic meaning of the text.
- **Storing in Vector Database**:
  - The generated vectors are stored in a specialized vector database (e.g., Pinecone, ChromaDB, Faiss).
  - This database enables efficient searching for similar content based on vector proximity.
- **Retrieval**: When a user inputs a query, the system uses techniques like cosine similarity or Euclidean distance to find the most relevant text chunks (vectors) from the database to use as context for the LLM.

### RAG Pipeline: Query Processing & Generation

**1. Query Processing Phase**
This stage activates after the vector database is prepared.

- **Query to Vector**: The user's text query is converted into a numerical vector using an embedding model.
- **Similarity Search**: This vector is used to search the vector database for the most relevant text chunks. This is done using techniques like cosine similarity or Euclidean distance.
- **Context Retrieval**: The best-matching text chunks are retrieved.
- **Context Enrichment (Augmentation)**: The retrieved context is enriched, often with metadata, to improve its quality and accuracy.
- **Prompt Creation**: A final prompt is assembled by combining the original user query with this enriched, retrieved context.

**2. Generation Phase**
- **LLM Input**: The combined prompt (original query + enriched context) is sent to a Large Language Model (LLM), such as those from OpenAI, Llama, or Google Gemini.
- **Final Output**: The LLM uses the provided context to generate a summarized, accurate, and human-like response.

The main benefit of this process is that the LLM's response is grounded in the specific, relevant information retrieved from the database, leading to more accurate and context-aware outputs.

---

## Model Distillation

### Definition
**Model Distillation** (also called **Knowledge Distillation**) is a technique for compressing a large, complex model (called the "teacher" model) into a smaller, more efficient model (called the "student" model) while preserving most of the teacher's performance.

### How It Works

1. **Teacher Model**: A large, high-performance model that has been trained on a dataset
2. **Student Model**: A smaller, lightweight model that learns from the teacher
3. **Knowledge Transfer**: The student model is trained to mimic the teacher's outputs, not just the original labels

### Key Concepts

**Soft Targets**: Instead of learning from hard labels (e.g., [0, 1, 0]), the student learns from the teacher's probability distributions (e.g., [0.1, 0.8, 0.1]). These "soft" probabilities contain more information about the relationships between classes.

**Temperature**: A parameter that controls how "soft" the probability distribution is. Higher temperature makes the distribution softer, revealing more information about what the teacher learned.

### Process

1. Train a large teacher model to high accuracy
2. Use the teacher to generate predictions (soft targets) on the training data
3. Train a smaller student model to match both:
   - The teacher's soft predictions
   - The original hard labels (optional)
4. The student learns to approximate the teacher's behavior

### Benefits

- **Reduced Model Size**: Student models are 10-100x smaller than teachers
- **Faster Inference**: Smaller models run faster, crucial for deployment
- **Lower Resource Requirements**: Can run on mobile devices or edge hardware
- **Maintained Performance**: Often retains 95%+ of teacher's accuracy

### Use Cases

- Deploying large language models on mobile devices
- Real-time applications requiring low latency
- Edge computing with limited resources
- Reducing costs in production environments

### Example

- **Teacher**: GPT-3 (175B parameters)
- **Student**: DistilBERT (66M parameters, 40% smaller than BERT)
- **Result**: 97% of BERT's performance, 60% faster

### Related Techniques

- **Pruning**: Removing unnecessary weights from a model
- **Quantization**: Reducing precision of model weights (e.g., float32 → int8)
- **Model Compression**: General term for making models smaller

---

## Review Questions for Next Class

1. What is the difference between encoding and embedding?
2. Why is one-hot encoding considered sparse?
3. What problem does TF-IDF solve that BOW doesn't?
4. How does GloVe differ from Word2Vec?
5. What is the advantage of contextual embeddings over static embeddings?
6. Why is BM25 considered a ranking algorithm rather than just an encoding method?
7. What are the three main types of tokenization strategies?
8. What special tokens are commonly used in text encoding and why?
9. What is model distillation and why is it useful?
10. What are "soft targets" in knowledge distillation?
