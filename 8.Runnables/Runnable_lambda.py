from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

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
    template="generate a joke on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_gen = RunnableSequence(prompt1, model, parser)

def word_counter(text: str) -> int:
    return len(text.split())

parallel_chain = RunnableParallel(
    {
        "joke" : RunnablePassthrough(),
        "count_word" : RunnableLambda(word_counter)
    }
)

final_chain = RunnableSequence(joke_gen, parallel_chain)

response = final_chain.invoke({'topic': 'train'})

print(response)