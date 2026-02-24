from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.environ['HUGGINGFACEHUB_ACCESS_TOKEN']

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HUGGINGFACEHUB_ACCESS_TOKEN
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="generate a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='explain the following joke - {text}',
    input_variables=['text']
)

parser = StrOutputParser()


joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explaination": RunnableSequence(prompt2, model, parser)
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

response = final_chain.invoke({'topic' : 'human'})

print(response)