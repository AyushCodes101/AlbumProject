import os
import sys
import logging
from Utils.dependencies import text_processor, vector_store, vectorizer

# Ensure logs are visible
logging.basicConfig(level=logging.INFO)

# Define a separate folder for the test FAISS index and metadata
TEST_INDEX_DIR = "test_faiss_data"
os.makedirs(TEST_INDEX_DIR, exist_ok=True)

# Set paths for the test FAISS index and metadata
test_index_path = os.path.join(TEST_INDEX_DIR, "test_faiss_index.index")
test_metadata_path = os.path.join(TEST_INDEX_DIR, "test_metadata.json")

# Reinitialize vector_store with separate test paths
vector_store.index_path = test_index_path
vector_store.metadata_path = test_metadata_path

# Ensure FAISS index is created if it doesn't exist
if not os.path.exists(test_index_path):
    logging.info("FAISS index not found. Creating a new one for testing.")
    vector_store.create_index(vectorizer.dimension)

def load_json_files(directory):
    """Load all JSON files from a given directory."""
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                json_files.append(f.read())
    return json_files

def process_and_store_data(json_files):
    """Process and store JSON data into the FAISS index."""
    chunks = []
    for json_data in json_files:
        chunks.extend(text_processor.process_json_data(json_data))

    if not chunks:
        logging.warning("No valid text chunks found in JSON files.")
        return

    logging.info(f"Processed {len(chunks)} text chunks")

    # Generate embeddings
    embeddings = vectorizer.create_embeddings(chunks)
    metadata = [{"chunk": chunk, "source": f"test_file_{i}"} for i, chunk in enumerate(chunks)]

    file_id = vector_store.get_next_file_id()
    vector_store.insert_records(embeddings, metadata, file_id)

def search_query(query):
    """Perform a search in FAISS and display results."""
    query_embedding = vectorizer.create_embeddings([query])[0]
    results = vector_store.search_index(query_embedding, k=5)

    print("\nSearch Results:")
    for res in results:
        print(f"Score: {res['score']:.4f} - {res['metadata']}")

if __name__ == "__main__":
    data_dir = input("Enter the directory path containing JSON files: ")
    json_files = load_json_files(data_dir)
    process_and_store_data(json_files)
    
    while True:
        query = input("Enter search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        search_query(query)
