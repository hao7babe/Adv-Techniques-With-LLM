from gensim.test.utils import datapath
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.cluster import KMeans
import numpy as np
import itertools

# Load the text8 dataset
dataset = api.load("text8")

# Hyperparameter options
win_size = [3, 7, 13, 25]
vector_size = [20, 70, 100, 300]

# Create combinations of hyperparameters
param_combinations = list(itertools.product(win_size, vector_size))

# Words to test with the model
words_for_clustering = ['yen', 'yuan', 'france', 'brazil', 'africa', 'asia']
transform_base_words = ['man', 'woman', 'daughter']

# Function to perform transformation and clustering
def perform_transform_and_clustering(model):
    try:
        # Transform: Embedding('man') - Embedding('woman') + Embedding('daughter')
        transform = model.wv['man'] - model.wv['woman'] + model.wv['daughter']

        # Find the most similar word to the result of the transform
        similar_word = model.wv.most_similar([transform], topn=1)[0][0]

        # Perform KMeans clustering on the specified words
        word_embeddings = np.array([model.wv[word] for word in words_for_clustering])
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(word_embeddings)
        clusters = dict(zip(words_for_clustering, kmeans.labels_))
        
        return similar_word, clusters

    except KeyError as e:
        return None, None

# Loop through each combination of parameters and store results
results = []
for win, size in param_combinations:
    # Train the Word2Vec model
    model = Word2Vec(sentences=dataset, vector_size=size, window=win, min_count=5, workers=4)
    
    # Perform transformation and clustering
    similar_word, clusters = perform_transform_and_clustering(model)
    
    # Store the results
    results.append({
        'win_size': win,
        'vector_size': size,
        'most_similar_word': similar_word,
        'cluster_labels': clusters
    })

# Convert results to a DataFrame for display
import pandas as pd
df_results = pd.DataFrame(results)
import ace_tools as tools; tools.display_dataframe_to_user(name="Word2Vec Results", dataframe=df_results)
