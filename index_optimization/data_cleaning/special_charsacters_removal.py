import string
import re

def remove_special_characters(text):
    cleaned_text = "".join([char for char in text if char not in string.punctuation])
    return cleaned_text

def remove_html_tags(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    return cleaned_text

input_text = "Hello, <b>world!</b> This is a test: does it remove <i>special characters</i> and HTML tags?"

text_without_html = remove_html_tags(input_text)
cleaned_text = remove_special_characters(text_without_html)

print("Original text:", input_text)
print("Text without HTML tags:", text_without_html)
print("Cleaned text:", cleaned_text)