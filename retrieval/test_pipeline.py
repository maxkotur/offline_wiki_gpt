from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# 1. Load documents
from data.test_documents import documents

# 2. Load embedding model on GPU
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

# 3. Create embeddings
vectors = embedder.encode(
    documents,
    batch_size=32,
    convert_to_numpy=True
)

# 4. Create FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# 5. Query
question = "Explain black holes"

query_vector = embedder.encode([question], convert_to_numpy=True)

distances, indices = index.search(query_vector, 3)

retrieved_docs = [documents[i] for i in indices[0]]

context = "\n\n".join(retrieved_docs)

# 6. Send to Ollama
response = ollama.chat(
    model="llama3",
    messages=[{
        "role": "user",
        "content": f"Use the context below to answer:\n\n{context}\n\nQuestion: {question}"
    }]
)

print("\n=== Retrieved Context ===")
print(context)

print("\n=== Model Answer ===")
print(response["message"]["content"])
