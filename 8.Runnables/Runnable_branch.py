from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda, RunnableBranch
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

prompt = PromptTemplate(
    template="create a report on this {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summarize the following {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

def word_count(text: str) -> int:
    return len(text.split())

topic_gen_chain = RunnableSequence(prompt, model, parser)

conditional_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(topic_gen_chain, conditional_chain)

response = final_chain.invoke({'topic' : 'deep ocear unknowns'})

print(response)