from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents" which are systems that can reason, learn, and act autonomously. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.
Machine learning is a subfield of AI that focuses on the use of data to improve performance on a specific task. Machine learning algorithms build a model based on sample data, which can then be used to make predictions or decisions without being explicitly programmed. Deep learning is a particular type of machine learning that utilizes artificial neural networks with multiple layers to learn from data.
Natural Language Processing (NLP) is another subfield of AI concerned with the interaction between computers and human language. NLP tasks include tasks like speech recognition, machine translation, sentiment analysis, and question answering.
AI has the potential to revolutionize many aspects of our lives, from the way we work to the way we interact with the world around us. However, there are also concerns about the potential risks of AI, such as job displacement and the development of autonomous weapons.
"""

chunk_size = 256
chunk_overlap = 20
separators = ["\n\n", "\n"]  # Split on double and single newlines

# Create text splitter object
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

docs = text_splitter.create_documents([text])

for doc in docs:
  print(doc)