from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

if not HUGGINGFACEHUB_API_TOKEN:
    print('not set')
    raise

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox("Select Algorithm", [ "XGBoost", "Gradient Boost", "Random Forest"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-friendly","Technical", "Code Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["short", "medium","large"])

template = load_prompt('template.json')

if st.button("summarize"):
    chain = template | model
    result = chain.invoke(
        {
            'paper_input' : paper_input,
            'style_input' : style_input,
            'length_input' : length_input
        }
    )
    st.write(result.content)