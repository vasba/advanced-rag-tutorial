import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt', quiet=True) #Supress nltk download log

def stem_text(text):
    stemmer = PorterStemmer()
    stemmed_text = " ".join([stemmer.stem(word) for word in text.split()])
    return stemmed_text

def normalize_text(text):
    lowercased_text = text.lower()
    stemmed_text = stem_text(lowercased_text)
    return stemmed_text

input_text = "Text normalization involves transforming text into a consistent format. This can include: Lowercasing, Stemming/Lemmatization."

normalized_text = normalize_text(input_text)

print("Original text:", input_text)
print("Normalized text:", normalized_text)