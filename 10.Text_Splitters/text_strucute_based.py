from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
In the previous article, we examined document loaders, which facilitate the loading of data from various document sources. There are various data loader classes for loading data from documents such as text files (TextLoader), PDF files (PyPDFLoader), web page data (WebBaseLoader), etcOnce the data is loaded from these sources, the amount of data is huge. We cannot feed this vast amount of data directly to an LLM or utilise it efficiently. Let us first understand what text splitting is and why we need it.
What is Text Splitting?Text splitting is the process of breaking a long document into smaller, easier-to-handle parts. Instead of giving the entire document to an AI system all at once — which might be too much to process — text splitting helps divide the content into chunks of a manageable size.These chunks are usually based on sentences, paragraphs, or character limits and sometimes include some overlap so the system doesn’t lose the meaning that flows from one part to the next.Press enter or click to view image in full sizeNeed of Splitting TextLLMs have a limited number of input tokens. Feeding an entire long document may exceed this limit.RAG systems rely on vector similarity search to retrieve relevant text chunks from a document database. Without splitting, the entire document is embedded as one large chunk, which reduces retrieval accuracy.Chunking also affects the embedding quality. Smaller chunks reduce semantic noise and improve the embedding quality.Now that we understand the need for text splitting, let us explore the different ways we can split the data.Ways to Split TextIn LangChain we have different TextSplitter classes to split the text. Firstly, install the required package using the command: pip install langchain_text_splitter . For the examples below, we have used text data from a sample PDF. So we use a PyPDFLoader document loader to load the data. And then perform splitting.Text Splitting based on LengthHere the splitting is done based on a predefined number of chunk_size . Below is an example.
We can directly split the Document objects using the method split_documents . Also, we can perform a split on text using the split_text method. Output is a list containing the resultant chunks.
Press enter or click to view image in full sizeOutput of Text Splitting based on Length


The chunk does not consist exactly of 100 characters because of what separator we have used. ext will be split only at new lines since we are using the new line (“\n”) as the separator. If any chunk has a size more than 100 but no new lines in it, it will be returned as such.
What is chunk_overlap ?chunk_overlap is one of the properties of the TextSplitter class (see in the code example above) that specifies the number of characters that are repeated between consecutive chunks when splitting text.
Become a memberThis helps in preserving the semantic meaning between two adjacent chunks by not creating a strict partition between them. Let us code one example for this.
Press enter or click to view image in full sizeOutput of chunk_overlap


Since we have set chunk_size=20 , 20 characters are shared between two consecutive chunks.Text-structured based
This is the most widely used kind of text splitter, so we will focus more on understanding this. As a base, this uses the fact that text data is organised in some hierarchical structure. The hierarchy can be represented as:The highest level is Paragraphs.Next comes Sentences that make a paragraph.Words combine to phrase a sentence.Words are made up of Characters.
We use RecursiveCharacterTextSplitter class in LangChain to split text recursively into smaller units, while trying to keep each chunk size in the given limit.How it works?Start from the highest level, e.g., paragraphs (separated by \n\n).If the paragraph is too long (exceeds the chunk_size), try to split it into sentences (e.g., using . or \n).If that’s still too large, go down to smaller units like words or characters.The splitting recurses downward until it produces chunks that are:Within the size limit (chunk_size)As semantically meaningful as possible.Let us code it now!
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])
print(chunks[-1])