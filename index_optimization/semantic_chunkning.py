import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = "{{OPENAI_API_KEY}}"  # Add your OpenAI API Key
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

breakpoint_threshold_type = "percentile"  # Options: "standard_deviation", "interquartile"

# Create the SemanticChunker with OpenAI Embeddings
text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(), breakpoint_threshold_type=breakpoint_threshold_type
)

text = """Galaxies are massive collections of stars, dust, and gas. The Milky Way galaxy, which contains our solar system, is estimated to contain hundreds of billions of stars. Galaxies come in a variety of shapes and sizes, including spiral, elliptical, and irregular galaxies.
Black holes are regions of spacetime with such intense gravity that nothing, not even light, can escape. They are formed when a massive star collapses in on itself. The gravity of a black hole is so strong that it can warp the fabric of spacetime around it.
The universe is constantly expanding and evolving. The Big Bang theory is the prevailing cosmological model for the universe. It states that the universe began with a very hot, dense state and has been expanding and cooling ever since. The exact cause of the Big Bang is still unknown."""

documents = text_splitter.create_documents([text])

print(documents)