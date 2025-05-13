import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the data
data = pd.read_csv('./data/features/combined_data.csv')

# Separate features and labels
X = data.iloc[:, 1:-1]  # Skip file name column and label column
y = data['source_file']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA Visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for label in set(y_encoded):
    plt.scatter(X_pca[y_encoded == label, 0], X_pca[y_encoded == label, 1], 
                label=label_encoder.inverse_transform([label])[0], alpha=0.6)
plt.title("PCA Clustering of Languages")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === t-SNE Visualization ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
for label in set(y_encoded):
    plt.scatter(X_tsne[y_encoded == label, 0], X_tsne[y_encoded == label, 1], 
                label=label_encoder.inverse_transform([label])[0], alpha=0.6)
plt.title("t-SNE Clustering of Languages")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
