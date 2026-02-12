from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    'Rakesh sharma is my father and he is shopkeeper, name of his shop is matrabhumi pan bhandar',
    'Rinkesh sharma is my mother and she is a teacher in santosh catholic high school',
    'Suryansh sharma is my brother and he is preparing for jee'
]

query = "who is Rinkesh"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

idx = np.argmax(np.array(scores))

print(query)
print(documents[idx])
print("similarity score is :", scores[idx])