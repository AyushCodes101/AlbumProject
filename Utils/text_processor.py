import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class TextProcessor:
    def process_json_data(self, json_data: Dict[str, Any]) -> List[str]:
        """Process any JSON data into text chunks"""
        try:
            # Flatten the JSON and extract all string values
            chunks = self._extract_strings(json_data)
            logger.info(f"Processed {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            raise

    def process_query(self, query: str) -> List[str]:
        """Process search query into chunks"""
        try:
            return self._chunk_text(query)
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise

    def _extract_strings(self, data: Any) -> List[str]:
        """Recursively extract all string values from JSON data"""
        chunks = []
        if isinstance(data, dict):
            for value in data.values():
                chunks.extend(self._extract_strings(value))
        elif isinstance(data, list):
            for item in data:
                chunks.extend(self._extract_strings(item))
        elif isinstance(data, str):
            chunks.extend(self._chunk_text(data))
        return chunks

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks of specified size"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]