from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

# sometimes the weak modols can't even understand pydantic output parser, use strong parser like 8B params
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
) 

model = ChatHuggingFace(llm=llm)

class FactSheet(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")

parser = PydanticOutputParser(pydantic_object=FactSheet)

template = PromptTemplate(
    template="give 3 facts about {topic}. \n\n {format_instructions}\n",
    input_variables=['topic'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
) 

chain = template | model | parser

result = chain.invoke({'topic': 'infinity'})

print(result)

# prompt = template.invoke({'topic' : 'infinity'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)