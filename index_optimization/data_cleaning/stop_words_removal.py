import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    cleaned_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return cleaned_text

input_text = "This is a simple example to demonstrate how the function removes stop words from the text."

cleaned_text = remove_stopwords(input_text)

print("Original text:", input_text)
print("Cleaned text:", cleaned_text)