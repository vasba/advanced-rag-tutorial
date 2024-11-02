import nltk
nltk.download('punkt', quiet=True) #Supress nltk download log

from langchain.text_splitter import NLTKTextSplitter

text = "This is an example text. It contains multiple sentences. Let's see how the splitter works."

text_splitter = NLTKTextSplitter()

sentences = text_splitter.split_text(text)

for sentence in sentences:
  print(sentence)