import os
import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv("../.env")

# below are read from .env
#os.environ['LANGCHAIN_TRACING_V2'] = 'true'
#os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
#os.environ['LANGCHAIN_API_KEY'] # read from .env
#os.environ['LANGCHAIN_PROJECT'] # read from .env

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths to the resource files
file_paths = [
    os.path.join(script_dir, "../resources/blog.langchain.dev_announcing-langsmith_.txt"),
    os.path.join(script_dir, "../resources/blog.langchain.dev_automating-web-research_.txt"),
]

# Load the documents
loaders = [TextLoader(path, encoding='UTF-8') for path in file_paths]

docs = []
for loader in loaders:
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=60)
splits = text_splitter.split_documents(docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """You are an AI language model assistant tasked with generating informative queries for a vector search engine.
The user has a question: "{question}"
Your goal is to create three variations of this question that capture different aspects of the user's intent. These variations will help the search engine retrieve relevant documents even if they don't use the exact keywords as the original question.
Provide these alternative questions, each on a new line.**
Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
  """ Unique union of retrieved docs """
  # Flatten list of lists
  flattened_docs = [doc for sublist in documents for doc in sublist]

  # Option 1: Check library documentation for hashable attribute (e.g., 'id')
  if hasattr(flattened_docs[0], 'page_content'):  # Replace 'id' with the appropriate attribute
      # Here we replaced it with 'page_content'
      unique_docs = list(set(doc.page_content for doc in flattened_docs))

  # Option 2: Convert to string (if suitable)
  else:
      unique_docs = list(set(str(doc) for doc in flattened_docs))

  return unique_docs

# Retrieve
question = "What is LangSmith, and why do we need it?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
len(docs)

# RAG
template = """Answer the following question based on this context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

answer = final_rag_chain.invoke({"question":question})

print(answer)
