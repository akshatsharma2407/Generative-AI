from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
# from langchain_core

import os

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate a short linkedin post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a short tweeter post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {                                       # CAN ALSO WRITE IT AS
        'linkedin' : prompt1 | model  | parser, # RunnableSequence(prompt1, model, parser)
        'tweet' : prompt2 | model | parser # RunnableSequence(prompt2, model, parser)
    }
)

response = parallel_chain.invoke({'topic' : 'AI'})

print(response)