from fastapi import APIRouter, HTTPException
from Utils.dependencies import text_processor, vectorizer, vector_store
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search")
async def search_image(query: str):
    try:
        # Ensure index is loaded
        if not vector_store.is_index_initialized():
            raise HTTPException(400, "FAISS index not initialized. Upload some data first.")
        
        # Process query
        query_chunks = text_processor.process_query(query)
        if not query_chunks:
            raise HTTPException(400, "Empty query after processing")
        
        # Generate embedding (use first chunk)
        query_embedding = vectorizer.create_embeddings([query_chunks[0]])[0]
        logger.info(f"Query embedding generated successfully.")
        
        # Search index
        results = vector_store.search_index(query_embedding)
        
        if not results:
            return {"message": "No matching records found"}
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(500, f"Search failed: {str(e)}")
