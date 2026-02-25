from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-iphone-17-black-256-gb/p/itm6eb39da622cdd'

loader = WebBaseLoader(url)

docs = loader.load()

template = PromptTemplate(template='answer the following {question} from {text}', input_variables=['question','text'])

chain = template | model | parser

response = chain.invoke({'question' : docs[0].page_content, 'text': 'some information about the storage, ram and camera'})

print(response)