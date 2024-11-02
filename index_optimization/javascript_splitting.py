from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

javascript_text = """
// Example: Sorting an Array

// Define an array of numbers
let numbers = [4, 2, 7, 1, 9];

// Sort the array in ascending order
numbers.sort((a, b) => a - b);

// Print the sorted array
console.log(numbers);
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=65, chunk_overlap=0
)

print(js_splitter.create_documents([javascript_text]))