# Word2Vec-Embeddings-on-Text8-Corpus
Trains a Word2Vec model on the Text8 corpus to generate word embeddings and visualizes selected word relationships using PCA, highlighting semantic similarities in a 2D space.

## 📋 Project Overview

This project demonstrates how to train a Word2Vec model from scratch using the Text8 corpus (a preprocessed Wikipedia dump). The model learns 50-dimensional word embeddings that capture semantic relationships between words, which are then visualized in 2D using Principal Component Analysis (PCA).

## 🎯 What Does This Do?

- **Trains** a Word2Vec model on ~17 million words from Wikipedia
- **Learns** semantic relationships and word meanings from context
- **Generates** 50-dimensional word embeddings
- **Visualizes** word relationships in an intuitive 2D scatter plot
- **Reveals** how semantically similar words cluster together

## 🔑 Key Features

- ✅ Uses complete Text8 corpus (~100MB of clean Wikipedia text)
- ✅ Trains Word2Vec with 50-dimensional embeddings
- ✅ Context window of 5 words for relationship learning
- ✅ Multi-threaded training for faster processing
- ✅ PCA dimensionality reduction for visualization
- ✅ Interactive scatter plot with word labels

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
Internet connection (first run only, to download Text8 corpus)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/notfakh/word2vec-text8-embeddings.git
cd word2vec-text8-embeddings
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

Run the script:
```bash
python word2vec_text8.py
```

**First Run:**
- Downloads Text8 corpus (~31MB compressed, ~100MB uncompressed)
- Takes 2-5 minutes depending on your machine
- Corpus is cached for future use

**Subsequent Runs:**
- Uses cached corpus
- Training takes 30-60 seconds

## 📊 Example Words Visualized

The script visualizes these words by default:
- **Royalty**: king, queen
- **Gender**: man, woman
- **Fruits**: apple, orange

### Expected Relationships:
- "king" and "queen" appear close together
- "man" and "woman" cluster nearby
- "apple" and "orange" form their own cluster
- Linear relationships visible (e.g., king-man ≈ queen-woman)

## 📈 Model Parameters

```python
Word2Vec(
    dataset,           # Text8 corpus (~17M words)
    vector_size=50,    # Embedding dimension
    window=5,          # Context window (5 words left/right)
    min_count=1,       # Include all words
    workers=4          # CPU cores for parallel training
)
```

### Parameter Explanation:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **vector_size** | 50 | Dimensionality of word vectors (balance between speed/quality) |
| **window** | 5 | Number of words considered for context on each side |
| **min_count** | 1 | Minimum word frequency (1 = include all words) |
| **workers** | 4 | Number of CPU threads for parallel training |

## 🔍 Text8 Corpus Details

**Dataset Information:**
- **Source**: First 100MB of Wikipedia dump (enwik9)
- **Size**: ~17 million words
- **Preprocessing**: Lowercased, special characters removed
- **Format**: Single file with space-separated words
- **Quality**: Clean, high-quality English text

**Why Text8?**
- Large enough for quality embeddings
- Small enough for quick training
- Standard benchmark in NLP research
- Pre-processed and ready to use

## 🛠️ Customization

### Visualize Different Words

Change the words to explore:
```python
words = ["dog", "cat", "animal", "pet", "furniture", "chair"]
```

### Increase Embedding Quality

Use higher dimensions:
```python
model = Word2Vec(dataset, vector_size=300, window=5, min_count=5, workers=4)
```

### Filter Rare Words

Exclude uncommon words:
```python
model = Word2Vec(dataset, vector_size=50, window=5, min_count=5, workers=4)
# min_count=5 means word must appear at least 5 times
```

### Adjust Context Window

Larger window captures broader relationships:
```python
model = Word2Vec(dataset, vector_size=50, window=10, min_count=1, workers=4)
```

## 🎨 Visualization Details

### PCA Dimensionality Reduction
- **Input**: 50-dimensional word vectors
- **Output**: 2-dimensional coordinates
- **Method**: Preserves maximum variance
- **Purpose**: Makes semantic relationships visible

### Scatter Plot Components
- **Points**: Each represents one word
- **Position**: Determined by semantic meaning
- **Labels**: Word annotations for identification
- **Distance**: Closer = more semantically similar

## 💡 Advanced Use Cases

### 1. Word Similarity
```python
# Find similar words
similar = model.wv.most_similar("king", topn=10)
print(similar)
# Output: [('queen', 0.85), ('prince', 0.82), ...]
```

### 2. Word Analogies
```python
# king - man + woman = ?
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(result)  # Should return 'queen'
```

### 3. Similarity Score
```python
# Calculate similarity between words
similarity = model.wv.similarity('king', 'queen')
print(f"Similarity: {similarity:.4f}")
```

### 4. Word Arithmetic
```python
# Vector operations
vector = model.wv['king'] - model.wv['man'] + model.wv['woman']
similar = model.wv.similar_by_vector(vector, topn=5)
```

## 🔬 Extending the Project

Ideas for enhancement:

1. **3D Visualization**
   ```python
   from mpl_toolkits.mplot3d import Axes3D
   pca = PCA(n_components=3)
   ```

2. **t-SNE Instead of PCA**
   ```python
   from sklearn.manifold import TSNE
   tsne = TSNE(n_components=2, random_state=42)
   ```

3. **Interactive Plotly Visualization**
   ```python
   import plotly.graph_objects as go
   # Create interactive 3D scatter plot
   ```

4. **Save and Load Model**
   ```python
   model.save("word2vec_text8.model")
   model = Word2Vec.load("word2vec_text8.model")
   ```

5. **Word Clustering**
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=5)
   clusters = kmeans.fit_predict(word_vectors)
   ```

## 📊 Performance Considerations

**Training Time:**
- CPU: 1-5 minutes (depending on cores)
- RAM: ~2GB required
- Storage: ~100MB for corpus cache

**Optimization Tips:**
- Increase `workers` parameter for faster training
- Use `min_count=5` to reduce vocabulary size
- Consider `vector_size=100` for better quality
- Use GPU with `gensim-gpu` for large-scale training

## 🤝 Contributing

Contributions welcome! Enhancement ideas:

- Add more visualization options (t-SNE, UMAP)
- Implement word analogy solver
- Create interactive web interface
- Add model evaluation metrics
- Include pre-trained model downloads
- Add word clustering analysis
- Compare different embedding sizes

## 👤 Author

**Fakhrul Sufian**
- GitHub: [@notfakh](https://github.com/notfakh)
- LinkedIn: [Fakhrul Sufian](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
- Email: fkhrlnasry@gmail.com

## 🙏 Acknowledgments

- Gensim library for Word2Vec implementation
- Tomas Mikolov et al. for the Word2Vec algorithm
- Matt Mahoney for creating the Text8 corpus
- Scikit-learn for PCA implementation

## 📚 References

- [Word2Vec Paper (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Text8 Corpus Information](http://mattmahoney.net/dc/textdata.html)
- [Understanding Word2Vec](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/)

## 🐛 Troubleshooting

**Issue: Slow training**
- Increase `workers` parameter
- Use smaller `vector_size`
- Increase `min_count` to reduce vocabulary

**Issue: Out of memory**
- Reduce `vector_size`
- Increase `min_count`
- Close other applications

**Issue: Download fails**
- Check internet connection
- Try again (downloads resume automatically)
- Manually download from [Text8 source](http://mattmahoney.net/dc/text8.zip)

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue in this repository
- Email: fkhrlnasry@gmail.com
- Connect on LinkedIn

---

⭐ If this project helped you understand Word2Vec and word embeddings, please give it a star!

## 🎓 Learning Outcomes

After working through this project, you'll understand:
- How Word2Vec learns word meanings from context
- Training embeddings on real Wikipedia data
- Semantic relationships in vector space
- Dimensionality reduction techniques (PCA)
- Visualizing high-dimensional NLP data
- Word similarity and analogy operations

**Perfect for:** NLP students, data scientists, and anyone interested in natural language processing!
