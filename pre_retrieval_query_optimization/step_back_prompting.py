import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain import hub

load_dotenv("../.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Few-shot learning for step-back prompting
examples = [
    {
        "input": "Did the Beatles ever write a book?",
        "output": "What types of creative works did the Beatles produce?"
    },
    {
        "input": "Was Albert Einstein a musician?",
        "output": "What fields did Albert Einstein work in?"
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Build the step-back prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

question_gen = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

question = "Did Leonardo da Vinci invent the printing press?"

question_gen.invoke({"question": question})

# Retrieve information

search = DuckDuckGoSearchAPIWrapper(max_results=4)

def retriever(query):
    results = search.run(query)
    return results

retriever(question)

retriever(question_gen.invoke({"question": question}))

# Build the RAG chain

response_prompt = hub.pull("langchain-ai/stepback-answer")

chain = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        "step_back_context": question_gen | retriever,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

answer = chain.invoke({"question": question})

print(answer)