from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
please summarize the machine learning algorithm named "{paper_input}" with following specifications:
explanation style : {style_input}
explanation length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical equations.
    - explain the mathematical concept using simple, intutive code
2. anologies:
    - Use relatable analogies to simplify complex ideas.
if certain information is not available to you, respond with "insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, aligned with provided style and length
""",
input_variables=['paper_input', 'style_input', 'length_input']  
)

template.save('template.json')