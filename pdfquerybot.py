# Impoert all necessary libraries
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv , find_dotenv
import os
from huggingface_hub import login
load_dotenv(find_dotenv())
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
login(token=os.getenv('HUGGINGFACE_ACCESS_TOKEN'))



# Step 1: Load PDF
pdfreader = PdfReader("Full_ML_Interview_Questions_Answers.pdf")  # Replace with your PDF path

# Extract text from PDF
# chunk pdf into pages and pages into text
raw_text = ""
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content



# Step 2: Split Text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function = len,
    separators = ["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_text(raw_text)
# raw_text
# Step 3: Convert Chunks into Embeddings using Open Source Model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Step 4: Store Embeddings into Chroma Vector Store (In-Memory)
vectorstore = Chroma.from_texts(texts, embedding)
# Step 5: Create Retriever to Fetch Relevant Chunks
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# STEP 6: Load LLM from HuggingFace (Gemma-3B)
llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-1b-it",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature= 0.7,
        max_new_tokens=100
    ),
)

model = ChatHuggingFace(llm=llm)



# Step 7: Create Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}
    """
)

# define parser
parser = StrOutputParser()



# define chain using all components
# Combine Prompt → LLM → OutputParser
chain = prompt | model | parser

# Step 8: Query Function
def ask_question(query):
    """
    Given a user query, retrieve relevant PDF chunks and generate an answer.
    """
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return chain.invoke({"context": context, "question": query})

# Example Query
ask_question("explain bias variance trade off")

ask_question("what is the bias how to handle it")

