from langchain.text_splitter import MarkdownTextSplitter

splitter = MarkdownTextSplitter(chunk_size=40, chunk_overlap=0)

markdown_text = """
# Unveiling Natural Language Processing

## Introduction to NLP

Dive into the fascinating world of Natural Language Processing (NLP), where computers analyze and understand human language. Explore the principles behind NLP algorithms and their applications in various domains.

### Sentiment Analysis

Discover how NLP techniques are used to analyze the sentiment of text data. From social media posts to customer reviews, learn how computers can discern the emotional tone of written content.

## Text Generation

Delve into the realm of text generation with NLP models. From autocomplete suggestions to creative storytelling, explore the algorithms that generate coherent and contextually relevant text.

### Language Translation

Unlock the power of language translation using NLP. Explore machine translation models that can convert text from one language to another with remarkable accuracy and fluency.

## Named Entity Recognition

Learn about Named Entity Recognition (NER) and how it extracts entities such as names, dates, and locations from text. Explore the applications of NER in information retrieval and knowledge extraction.

### Document Summarization

Master the art of document summarization with NLP. Discover algorithms that condense lengthy texts into concise summaries, extracting the most important information while preserving context.

## Text Classification

Explore the field of text classification and its role in organizing and categorizing textual data. From spam detection to sentiment analysis, uncover the algorithms that classify text into predefined categories.
"""

print(splitter.create_documents([markdown_text]))