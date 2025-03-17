import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self):
        self.model = None
        self.dimension = 384  # default for all-MiniLM-L6-v2
        self._load_model()

    def _load_model(self):
        """Load embedding model"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            if not texts:
                return []
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise