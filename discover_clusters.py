import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model
import hdbscan
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load Data and the Trained Model ---

def one_hot_encode(sequence, max_len=600):
    """One-hot encodes a DNA sequence to a fixed length."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded = np.zeros((max_len, len(mapping)))
    padded_seq = sequence[:max_len].upper().ljust(max_len, 'N')
    for i, base in enumerate(padded_seq):
        if base in mapping:
            encoded[i, mapping[base]] = 1
    return encoded

print("Loading the full labeled dataset...")
df = pd.read_csv('fungi_labeled_data.csv')
df.dropna(subset=['family'], inplace=True)
df = df[df['family'] != 'Unknown_Family'].copy()
print(f"Loaded {len(df)} total sequences to be clustered.")

print("Loading the trained CNN model using load_model...")
trained_model = load_model('fungi_family_classifier.h5')


# --- 2. Create the Embedding Model by Name ---

print("Creating embedding model by accessing the named layer...")
# This is the foolproof method: get the model's input and the output of our named layer
embedding_model = Model(
    inputs=trained_model.input,
    outputs=trained_model.get_layer('embedding_layer').output
)
embedding_model.summary()


# --- 3. Generate Embeddings for ALL Sequences ---

print("\nPreprocessing all sequences for embedding...")
X_full = np.array(df['sequence'].apply(one_hot_encode).tolist())

print("Generating embeddings (genetic fingerprints)...")
embeddings = embedding_model.predict(X_full)
print(f"Shape of embeddings: {embeddings.shape}")


# --- 4. Unsupervised Clustering with HDBSCAN ---

print("\nStarting unsupervised clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(embeddings)

df['cluster'] = cluster_labels
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"\n--- Discovery Complete! ---")
print(f"Found {num_clusters} distinct clusters (species/OTUs).")

df.to_csv('fungi_final_clusters.csv', index=False)
print("Final results saved to 'fungi_final_clusters.csv'")


# --- 5. Visualize the Clusters ---

print("\nGenerating 2D visualization of clusters...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

df_plot = pd.DataFrame(embedding_2d, columns=['UMAP_1', 'UMAP_2'])
df_plot['cluster'] = cluster_labels

plt.figure(figsize=(16, 12))
unique_clusters = sorted(df_plot['cluster'].unique())
palette = sns.color_palette("hsv", n_colors=len(unique_clusters))

sns.scatterplot(
    x='UMAP_1', y='UMAP_2',
    hue='cluster',
    palette=palette,
    data=df_plot,
    legend='full',
    alpha=0.7,
    s=50
)

plt.title('2D Visualization of Discovered Genetic Clusters', fontsize=16)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('cluster_visualization.png')
print("Cluster visualization saved as 'cluster_visualization.png'")