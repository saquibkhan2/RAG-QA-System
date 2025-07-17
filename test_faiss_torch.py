import torch
import faiss
import numpy as np

print('torch:', torch.__version__)
print('faiss:', faiss.__version__)

# Minimal FAISS test
vecs = np.random.rand(10, 4).astype('float32')
index = faiss.IndexFlatL2(4)
index.add(vecs)
D, I = index.search(np.random.rand(1, 4).astype('float32'), 3)
print('FAISS search result:', I) 