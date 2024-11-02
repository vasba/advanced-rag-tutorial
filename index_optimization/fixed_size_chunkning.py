from langchain.text_splitter import CharacterTextSplitter

page_content = """The quick brown fox jumps over the lazy dog. In computing, a text file
is a computer file that stores characters in a format that is readable by humans.
Text files are created using a text editor.  There are two main types of text
files: plain text files and formatted text files. Plain text files contain
only printable characters, such as letters, numbers, and symbols. Formatted
text files may contain additional characters, such as control characters,
that are used to format the text. These control characters may not be
displayed when the text file is viewed in a text editor.

Common file extensions for plain text files include .txt, .text, and .asc.
Common file extensions for formatted text files include .doc, .docx, .odt,
and .rtf."""

sentences = page_content.split("\n")

chunk_size = 256
chunk_overlap = 20

text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Split each sentence/paragraph into chunks
all_chunks = []
for sentence in sentences:
  chunks = text_splitter.create_documents([sentence])
  all_chunks.extend(chunks)  # Add chunks from each sentence to a single list

for doc in all_chunks:
  print(doc)