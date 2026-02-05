from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from ..mcp.handlers import MemoryHandlers
from ..config.logging import get_logger

logger = get_logger(__name__)


class AddMemoryRequest(BaseModel):
    """Request model for adding memory."""
    content: str


class SearchMemoryRequest(BaseModel):
    """Request model for searching memory."""
    query: str


class ListMemoryRequest(BaseModel):
    """Request model for listing memories."""
    limit: Optional[int] = 10
    offset: Optional[int] = 0


class HTTPServer:
    """HTTP server for memory operations."""
    
    def __init__(self, handlers: MemoryHandlers):
        """
        Initialize HTTP server.
        
        Args:
            handlers: Memory handlers instance
        """
        self.handlers = handlers
        self.app = FastAPI(title="EzMemory HTTP Server", version="1.0.0")
        self._register_routes()
    
    def _register_routes(self):
        """Register HTTP routes."""
        
        @self.app.post("/api/add_memory", response_model=dict)
        async def add_memory(request: AddMemoryRequest):
            """
            Add a new memory.
            
            This endpoint stores new memories about the user, their preferences,
            or any relevant information that can be useful in future conversations.
            """
            try:
                result = await self.handlers.add_memory(request.content)
                return result
            except Exception as e:
                logger.error(f"Error adding memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/search_memory", response_model=dict)
        async def search_memory(request: SearchMemoryRequest):
            """
            Search for relevant memories based on a query.
            
            Use this to find information that was previously stored in memory.
            Example queries: "What did I tell you about the weather last week?"
            or "What did I tell you about my friend John?"
            """
            try:
                result = await self.handlers.search_memory(request.query)
                return result
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/list_memory", response_model=dict)
        async def list_memory(request: ListMemoryRequest):
            """
            List all stored memories with pagination.
            
            Returns a paginated list of all memories in the collection.
            """
            try:
                result = await self.handlers.list_memory(
                    limit=request.limit,
                    offset=request.offset
                )
                return result
            except Exception as e:
                logger.error(f"Error listing memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/delete_all_memory", response_model=dict)
        async def delete_all_memory():
            """
            Delete all memories from the collection.
            
            This will permanently remove all stored memories in the current collection.
            Use with caution as this operation cannot be undone.
            """
            try:
                result = await self.handlers.delete_all_memory()
                return result
            except Exception as e:
                logger.error(f"Error deleting all memories: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok", "service": "EzMemory HTTP Server"}
    
    def get_app(self):
        """Get the FastAPI application."""
        return self.app
