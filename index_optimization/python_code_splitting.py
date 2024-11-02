from langchain.text_splitter import PythonCodeTextSplitter

python_text = """
# Example: Fibonacci Sequence

def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        next_num = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_num)
    return fib_sequence

fib_numbers = fibonacci(10)
print(fib_numbers)
"""

python_splitter = PythonCodeTextSplitter(chunk_size=100, chunk_overlap=0)
print(python_splitter.create_documents([python_text]))