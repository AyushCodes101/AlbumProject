from fastapi import FastAPI
from Routers.upload import router as upload_router
from Routers.search import router as search_router
import logging

# Log Path
log_file_path = "app.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Log to Console
    ]
)

app = FastAPI(title="Image Search API")

# Include routers
app.include_router(upload_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy"}