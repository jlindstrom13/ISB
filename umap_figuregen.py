# Using umap_stability.py output file (umap_embedding.npy) create a graph

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


embedding = np.load("umap_embedding.npy")

plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    alpha = 0.1,
    s = 2)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Stability dataset', fontsize=15);
plt.savefig("umap_projection_full_2")