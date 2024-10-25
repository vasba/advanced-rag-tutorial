# Task 1: Import the Libraries
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time

# Task 2(b): Call load_dotenv() to load environment variables
load_dotenv()

# Task 6: Define Utility Functions
def stream_data(content):
    for word in content.split(" "):
        yield word + " "
        time.sleep(0.02)

def get_vectorstore_from_url(url):
    # Load the web page content
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create a vector store from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    return vector_store

# Task 8: Create Chains for Context and Conversation
def get_context_retriever_chain(vector_store):
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Get the retriever from the vector store
    retriever = vector_store.as_retriever()
    
    # Define the prompt for the retriever
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    # Create the retriever chain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Define the prompt for the conversational RAG chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # Create the document combination chain
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Task 9(a): Process User Queries and Update Chat
def get_response(user_input):
    # Create the retriever and conversational RAG chains
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # Invoke the conversational RAG chain with the current chat history and user input
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# Task 3: Create a Streamlit Application
# Define the main function to initialize the Streamlit app
def main():
    # Set the title of the app
    st.title("Chat with Webpage!")
    # Task 4: Add a Text Input for URL
     # Create a placeholder for the text input
    text_input_container = st.empty()
    
    # Add a text input field for the user to enter a web page URL
    website_url = text_input_container.text_input("Webpage URL:")

    # Task 5: Display Information Message for URL Input
    # Check if the URL is empty and display an info message if true
    if website_url is None or website_url == "":
        st.info("Please enter a webpage URL")
    # Task 7: Initialize Session State
    else:
        # Clear the text input container once the URL is provided
        text_input_container.empty()
        
        # Initialize chat history if not already present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?"),
            ]
        
        # Initialize vector store if not already present
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
        
        # Task 9(b): Process User Queries and Update Chat
        # Get user input and process the query
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            # Append user query to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            # Get the response from the model
            response = get_response(user_query)
            
            # Append the response to chat history
            if response:
                st.session_state.chat_history.append(AIMessage(content=response))

        # Display the conversation
        for idx, message in enumerate(st.session_state.chat_history):
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    if idx == len(st.session_state.chat_history) - 1:
                        content = message.content
                        st.write_stream(stream_data(content))
                    else:
                        st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    pass

# Run the main function when the script is executed
if __name__ == "__main__":
    main()