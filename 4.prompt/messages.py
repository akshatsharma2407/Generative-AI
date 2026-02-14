from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are creative person who see the world in very very different manner'),
    HumanMessage(content='what do you think about human')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)