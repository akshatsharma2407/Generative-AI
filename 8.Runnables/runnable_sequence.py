from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser) # same as pipe operator 

result = chain.invoke({'topic' : 'AI'})

print(result)