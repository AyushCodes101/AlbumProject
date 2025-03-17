from fastapi import APIRouter, File, UploadFile, HTTPException
from Utils.dependencies import text_processor, vectorizer, vector_store
import json
import logging
from typing import List
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_image_data(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(400, "No files uploaded")

        results = []
        total_chunks = 0

        # Ensure FAISS index
        if vector_store.index is None:
            try:
                vector_store.create_index(vectorizer.dimension)
            except ValueError:
                pass  # Index already exists

        async def process_file(file: UploadFile):
            """Processes a single file"""
            try:
                contents = await file.read()
                json_data = json.loads(contents)

                # Ensure valid JSON structure
                if not isinstance(json_data, (dict, list)):
                    return {"file": file.filename, "status": "failed", "error": "Expected JSON object or array"}

                # Process JSON data
                chunks = text_processor.process_json_data(json_data)
                if not chunks:
                    return {"file": file.filename, "status": "skipped", "error": "No valid text chunks"}

                # Generate embeddings
                embeddings = vectorizer.create_embeddings(chunks)

                # Insert into FAISS index
                file_id = vector_store.get_next_file_id()
                metadata = [{"chunk": chunk, "source": file.filename} for chunk in chunks]
                vector_store.insert_records(embeddings, metadata, file_id)

                return {"file": file.filename, "status": "success", "chunks_processed": len(chunks)}

            except json.JSONDecodeError:
                return {"file": file.filename, "status": "failed", "error": "Invalid JSON format"}
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                return {"file": file.filename, "status": "failed", "error": str(e)}

        # Process all files **concurrently**
        file_processing_tasks = [process_file(file) for file in files]
        results = await asyncio.gather(*file_processing_tasks)

        total_chunks = sum(res.get("chunks_processed", 0) for res in results if res["status"] == "success")

        return {
            "message": f"Successfully processed {len(files)} files with {total_chunks} total chunks",
            "results": results
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, str(e))
