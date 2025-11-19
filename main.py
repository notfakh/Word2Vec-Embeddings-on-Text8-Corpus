import gensim.downloader as api
from gensim.models import Word2Vec

# Load the Text8 corpus
dataset = api.load("text8")

# Train the Word2Vec model
model = Word2Vec(dataset, vector_size=50, window=5, min_count=1, workers=4)

# Now you can use this model for word embeddings
words = ["king", "queen", "man", "woman", "apple", "orange"]
word_vectors = [model.wv[word] for word in words]

# PCA for visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
plt.show()
