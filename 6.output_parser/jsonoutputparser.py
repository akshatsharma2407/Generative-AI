from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="give me 5 shayri of topic but write in hinglish language - {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic" : 'sad'})

print(result)