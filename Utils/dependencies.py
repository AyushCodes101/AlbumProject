from Utils.text_processor import TextProcessor
from Utils.vectorizer import Vectorizer
from Utils.vector_store import VectorStore

text_processor = TextProcessor()
vectorizer = Vectorizer()
vector_store = VectorStore(index_path="faiss_index.index", metadata_path="metadata.json")