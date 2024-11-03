import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv("../.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Load and split documents

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

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Generate embeddings (multiple)

multi_llm = OpenAI(n=3, best_of=4)

embeddings = HypotheticalDocumentEmbedder.from_llm(multi_llm, OpenAIEmbeddings(), "web_search")

query = "What is LangSmith, and why do we need it?"

result = embeddings.embed_query(query)

# Query the vector store for HyDE

vectorstore.similarity_search(query)

# Generate a hypothetical document

system = """
As a knowledgeable and helpful research assistant, your task is to provide informative answers based on the given context.
Use your extensive knowledge base to offer clear, concise, and accurate responses to the user's inquiries.
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

context = prompt | llm | StrOutputParser()

answer = context.invoke(
    {
        "What is LangSmith, and why do we need it?"
    }
)

print(answer)

# Return the hypothetical document and original question

chain = RunnablePassthrough.assign(hypothetical_document=context)

chain.invoke(
    {
        "question": "What is LangSmith, and why do we need it?"
    }
)

