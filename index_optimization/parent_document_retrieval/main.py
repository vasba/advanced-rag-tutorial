import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv("../.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

embeddings = OpenAIEmbeddings()

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

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

full_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter
)

full_doc_retriever.add_documents(docs)

print(list(store.yield_keys()))  # List document IDs in the store

sub_docs = vectorstore.similarity_search("What is LangSmith?", k=2)
print(len(sub_docs))

print(sub_docs[0].page_content)  

retrieved_docs = full_doc_retriever.invoke("What is LangSmith?")

print(len(retrieved_docs[0].page_content)) 
print(retrieved_docs[0].page_content)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)  
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)   

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=OpenAIEmbeddings()
)

store = InMemoryStore()

big_chunks_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Adding documents
big_chunks_retriever.add_documents(docs)
print(len(list(store.yield_keys())))  # List document IDs in the store

sub_docs = vectorstore.similarity_search("What is LangSmith?", k=2)
print(len(sub_docs))

print(sub_docs[0].page_content)  

retrieved_docs = big_chunks_retriever.invoke("What is LangSmith?")
print(len(retrieved_docs))

print(len(retrieved_docs[0].page_content)) 
print(retrieved_docs[0].page_content)  

qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                chain_type="stuff",
                                retriever=big_chunks_retriever)

query = "What is LangSmith?"

response = qa.invoke(query)
print(response)

