# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
import os

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

class Review(BaseModel):
    key_themes: Annotated[list[str], Field(description='write the default theme discuessed in this review', examples=["Samsung", "Processor", "Charger"])]
    summary: Annotated[str, Field(description="a bried summary of review")]
    sentiment: Annotated[Literal["pos","neg","neutral"], Field(description="Return sentiment of review either negative, positive or neutral")]
    pros: Annotated[Optional[list[str]], Field(description="write all the pros inside a list, keep it blank if not found", default=[None])]
    cons: Annotated[Optional[list[str]], Field(description="write all the cons inside a list, keep it blank if not found", default=[None])]
    name: Annotated[Optional[str], Field(description="write name of reviewer, keep it blank if not found", default=None)]
 

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
In midnight black, this sleek device arrives as a testament to human craft and drive.
The A16 Bionic processor beats its drum with six gigabytes of memory, second to none, 
whilst one hundred-twenty-eight gigabytes of storage await to hold your moments and grace. 
The forty-eight megapixel eye captures photography with precision and ease, and the Super Retina screen glows with two thousand nits of perfect light. 
The Dynamic Island, smart and true, displays alerts where the old notch once flew. This silicon dream breaks free with USB-C, abandoning the corded past, whilst aerospace aluminum and reinforced glass stand resolute against dust and water. Eighty hours of listening bliss and twenty hours of video's kiss flow from this device, powered by the A16—the finest processor ever conceived. At its worthy price, this iPhone 15 in black's embrace stands supreme, a reality refined and a digital dream materialized before your eyes. An unequivocal masterpiece of technological magnificence!
""")

print(result)