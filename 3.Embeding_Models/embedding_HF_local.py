from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# text = "Delhi is capital of India"

document = [
    "delhi is capital of India",
    "kolkata is capital of wb",
    "paris is capital of France"
]

vector = embedding.embed_documents(document)

print(str(vector))