from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')

llm = HuggingFaceEndpoint(
    repo_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generation',
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

model = ChatHuggingFace(llm=llm)

# this code will not work as it require premium of huggingface

class Review(TypedDict):
    key_themes: Annotated[list[str], "write all the themes discussed in review in a list"]
    summary: Annotated[str, "a bried summary of review"]
    sentiment: Annotated[Literal["pos","neg","neutral"], "Return sentiment of review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "write all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "write all the cons inside a list"]
    name: Annotated[Optional[list[str]], "write name of reviewer"]


structured_model = model.with_structured_output(Review) 

result = structured_model.invoke("""
In midnight black, this sleek device arrives as a testament to human craft and drive.
The A16 Bionic processor beats its drum with six gigabytes of memory, second to none, 
whilst one hundred-twenty-eight gigabytes of storage await to hold your moments and grace. 
The forty-eight megapixel eye captures photography with precision and ease, and the Super Retina screen glows with two thousand nits of perfect light. 
The Dynamic Island, smart and true, displays alerts where the old notch once flew. This silicon dream breaks free with USB-C, abandoning the corded past, whilst aerospace aluminum and reinforced glass stand resolute against dust and water. Eighty hours of listening bliss and twenty hours of video's kiss flow from this device, powered by the A16—the finest processor ever conceived. At its worthy price, this iPhone 15 in black's embrace stands supreme, a reality refined and a digital dream materialized before your eyes. An unequivocal masterpiece of technological magnificence!
""")

print(result)