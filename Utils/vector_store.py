import os
import faiss
import numpy as np
import json
import logging
from typing import Dict, List, Optional
import threading

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str = "faiss_index.index", metadata_path: str = "metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.Index] = None
        self.metadata_store: Dict[int, Dict] = {}
        self.dimension: Optional[int] = None
        self.lock = threading.Lock()
        self.file_counter = 0  # Track unique file IDs  

        # Load existing index and metadata
        self._load_index()
        self._load_metadata()

    def _load_index(self):
        """Load FAISS index from disk if it exists"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                logger.info(f"Loaded FAISS index from {self.index_path} with dimension {self.dimension}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {str(e)}")
                self.index = None  # Prevent crashes by setting it to None

    def _load_metadata(self):
        """Load metadata from disk if it exists"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata_store = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_path}")
            except Exception as e:
                logger.error(f"Failed to load metadata: {str(e)}")
                self.metadata_store = {}  # Prevent crashes by setting it to an empty dict

    def is_index_initialized(self) -> bool:
        """Check if the FAISS index has been initialized"""
        return self.index is not None and self.index.ntotal > 0

    def _save_index(self):
        """Save FAISS index to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")

    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata_store, f)
            logger.info(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")

    def create_index(self, dimension: int) -> None:
        """Create a new FAISS index"""
        with self.lock:
            try:
                if self.index is not None:
                    raise ValueError("Index already exists")
                self.index = faiss.IndexFlatL2(dimension)
                self.dimension = dimension
                logger.info(f"Created new index with dimension {dimension}")
                self._save_index()
            except Exception as e:
                logger.error(f"Index creation failed: {str(e)}")
                raise

    def insert_records(self, embeddings: List[List[float]], metadata: List[Dict], file_id: int) -> None:
        """Insert multiple records into the index"""
        with self.lock:
            try:
                if self.index is None:
                    raise ValueError("Index not initialized")
                
                vectors = np.array(embeddings, dtype='float32')
                start_id = self.index.ntotal
                self.index.add(vectors)
                
                for i, meta in enumerate(metadata):
                    self.metadata_store[start_id + i] = {
                        **meta,
                        "file_id": file_id  # Add file ID to metadata
                    }
                logger.info(f"Inserted {len(embeddings)} records for file ID {file_id}")
                
                # Save index and metadata to disk
                self._save_index()
                self._save_metadata()
            except Exception as e:
                logger.error(f"Record insertion failed: {str(e)}")
                raise

    def search_index(self, query_embedding: List[float], k: Optional[int] = None) -> List[Dict]:
        """Search the index for similar vectors"""
        try:
            if self.index is None:
                raise ValueError("Index not initialized")
            
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty. No data to search.")
                return []
            
            # Set k to the total number of indexed items if not specified
            k = self.index.ntotal if k is None else k
            
            query = np.array([query_embedding], dtype='float32')
            distances, indices = self.index.search(query, k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx in self.metadata_store:
                    results.append({
                        "score": float(distance),
                        "metadata": self.metadata_store[idx]
                    })
            logger.info(f"Found {len(results)} search results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def get_next_file_id(self) -> int:
        """Generate a unique file ID"""
        with self.lock:
            self.file_counter += 1
            return self.file_counter
