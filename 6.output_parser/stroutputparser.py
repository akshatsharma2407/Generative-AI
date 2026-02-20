from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="write a detailed summary about the {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="write a 1 liner summary of following text. \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'about the infinity'})

print(result)

# prompt1 = template1.invoke({"topic": "me or mere tanhai aksar ye bate kia krte hai kash tu hota to esa hota, kash tu hota to vesa hota"})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({"text" : result})

# final_result = model.invoke(prompt2)

# print(final_result.content)