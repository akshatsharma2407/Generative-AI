from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('documents/directory/delivery_partners.csv')

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)