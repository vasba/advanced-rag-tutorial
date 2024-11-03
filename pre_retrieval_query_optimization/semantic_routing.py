import os
from dotenv import load_dotenv
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv("../.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Define domain-specific prompts and questions
personal_finance_template = """You are a personal finance expert with extensive knowledge of budgeting, investing, and financial planning. You offer clear and practical advice on managing money and making sound financial decisions.

Here is a question:
{query}"""

book_review_template = """You are an experienced book critic with extensive knowledge of literature, genres, and authors. You provide thoughtful and analytical reviews and insights about books.

Here is a question:
{query}"""

health_fitness_template = """You are a certified health and fitness expert with a deep understanding of nutrition, exercise routines, and wellness strategies. You offer practical and evidence-based advice about health and fitness.

Here is a question:
{query}"""

travel_guide_template = """You are a seasoned travel expert with extensive knowledge of destinations, travel tips, and cultural insights. You provide detailed and useful advice about travel.

Here is a question:
{query}"""

personal_finance_questions = [
    "What are the best strategies for saving money?",
    "How do I start investing in the stock market?",
    "What should I consider when creating a budget?",
]

book_review_questions = [
    "What makes a novel a classic?",
    "How do you analyze the themes in a book?",
    "What are the key differences between literary fiction and genre fiction?",
]

health_fitness_questions = [
    "What are the benefits of a balanced diet?",
    "How often should I exercise to maintain good health?",
    "What are effective strategies for losing weight?",
]

travel_guide_questions = [
    "What are the must-see attractions in Tokyo?",
    "How can I travel on a budget?",
    "What should I know before traveling to a foreign country?",
]

# Create text embeddings#

embeddings = OpenAIEmbeddings()

book_review_question_embeddings = embeddings.embed_documents(book_review_questions)
health_fitness_question_embeddings = embeddings.embed_documents(health_fitness_questions)
travel_guide_question_embeddings = embeddings.embed_documents(travel_guide_questions)
personal_finance_question_embeddings = embeddings.embed_documents(personal_finance_questions)

# Route based on maximum similarity

def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    book_review_similarity = cosine_similarity([query_embedding], book_review_question_embeddings)[0]
    health_fitness_similarity = cosine_similarity([query_embedding], health_fitness_question_embeddings)[0]
    travel_guide_similarity = cosine_similarity([query_embedding], travel_guide_question_embeddings)[0]
    personal_finance_similarity = cosine_similarity([query_embedding], personal_finance_question_embeddings)[0]

    max_similarity = max(
        max(book_review_similarity), 
        max(health_fitness_similarity), 
        max(travel_guide_similarity), 
        max(personal_finance_similarity)
    )

    if max_similarity == max(book_review_similarity):
        print("Using BOOK REVIEW")
        return PromptTemplate.from_template(book_review_template)
    elif max_similarity == max(health_fitness_similarity):
        print("Using HEALTH AND FITNESS")
        return PromptTemplate.from_template(health_fitness_template)
    elif max_similarity == max(travel_guide_similarity):
        print("Using TRAVEL GUIDE")
        return PromptTemplate.from_template(travel_guide_template)
    else:
        print("Using PERSONAL FINANCE")
        return PromptTemplate.from_template(personal_finance_template)

# Utilize the selected prompt

input_query = {"query": "What are effective strategies for losing weight?"}
prompt = prompt_router(input_query)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

input_query = {"query": "What are the must-see attractions in USA?"}

response = chain.invoke(input_query["query"])

print(response)
