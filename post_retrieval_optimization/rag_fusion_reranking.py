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

# Index documents

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG-Fusion: Query generation

template = """You are an AI language model assistant tasked with generating seach queries for a vector search engine.
The user has a question: "{question}"
Your goal/task is to create five variations of this {question} that capture different aspects of the user's intent. These variations will help the search engine retrieve relevant documents even if they don't use the exact keywords as the original question.
Provide these alternative questions, each on a new line.**
Original question: {question}"""

rag_fusion_prompt_template = ChatPromptTemplate.from_template(template)

generate_queries = (
    rag_fusion_prompt_template
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# Retrieval with reciprocal rank fusion (RRF)

def reciprocal_rank_function(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a unique string identifier 
            doc_str = str(doc)  # Simple string conversion 
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (doc, score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

question = "What is LangSmith, and why do we need it?"
retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_function
docs = retrieval_chain.invoke({"question":question})
len(docs)

# Run the RAG model

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