from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class sentiment(BaseModel):
    sentiment : Literal["Positive", "Negative"] = Field(description="Write sentiment of feedback either Negative or Positive", examples=["Positive", "Negative"])

parser = StrOutputParser()

pydantic_parser = PydanticOutputParser(pydantic_object=sentiment)

prompt1 = PromptTemplate(
    template="Classify the sentiment as 'positive' or 'negative' \n {feedback} \n\n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions' : pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser 

prompt2 = PromptTemplate(
    template="you are a customer chatbot, write a appropriate small response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="you are a customer chatbot, write a appropriate small response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)
 
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "count not find sentiment")
)

chain = classifier_chain | branch_chain

response = chain.invoke({'feedback' : 'This is sigma phone'})
 
print(response)