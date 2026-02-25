from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text_splitter = SemanticChunker(
    embedding, breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1 
)

text = """Climate change is driven largely by human activities such as burning fossil fuels and deforestation. These actions increase greenhouse gases in the atmosphere, trapping more heat.

As temperatures rise, we are seeing more frequent heatwaves, stronger storms, and melting glaciers. These changes threaten ecosystems, agriculture, and coastal communities.

To reduce the impact, countries are investing in renewable energy like solar and wind. Individuals can also help by conserving energy, reducing waste, and using public transportation.

Scientists continue to research long-term solutions, but meaningful change will require cooperation between governments, businesses, and citizens."""

docs = text_splitter.create_documents([text])

print(len(docs))
print('*'*50)
print(docs)