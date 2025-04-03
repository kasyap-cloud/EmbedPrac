from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

paragraph = """Machine learning is a subset of artificial intelligence that enables 
systems to learn patterns from data and make decisions without being explicitly programmed."""

embedding = model.encode(paragraph)

print("Embedding Vector:", embedding)
print("Shape:", embedding.shape) 
print("max", np.argmax(embedding))