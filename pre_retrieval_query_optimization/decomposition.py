import os
import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv("../.env")

loader = WebBaseLoader(
    web_paths=("https://blog.langchain.dev/announcing-langsmith",),
)

blog_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=20)

splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# Create sub-questions: Decompose the main question

template = """You are an AI language model assistant that generates multiple sub-questions related to an input question. \n
Your task is to break down the input into three sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Original question: {question}"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

question = "What is LangSmith, and why do we need it?"

generate_queries_decomposition.invoke({"question":question})

# Generate answers for sub-questions
prompt_rag = hub.pull("rlm/rag-prompt")

sub_questions = generate_queries_decomposition.invoke({"question":question})
rag_results = []
for sub_question in sub_questions:
  retrieved_docs = retriever.invoke(sub_question)
  answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs,
                                                                "question": sub_question})
  rag_results.append(answer)

# Merge sub-questions and answers
def format(questions, answers):
    """Format Q and A"""

    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

context = format(sub_questions, rag_results)

# Generate the final answer
template = """Here is a set of Q and A:
{context}
Use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

answer = final_rag_chain.invoke({"context":context,"question":question})

print(answer)