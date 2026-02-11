from fastmcp import FastMCP
from typing import Any, Optional
from ..config.logging import get_logger
from .handlers import MemoryHandlers
from typing import Annotated


logger = get_logger(__name__)


class MemoryMCPServer:
    """MCP server for memory operations."""
    
    def __init__(self, handlers: MemoryHandlers):
        """
        Initialize MCP server.
        
        Args:
            handlers: Memory handlers instance
        """
        self.handlers = handlers
        self.mcp = FastMCP[Any]("EzMemory")
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.mcp.tool()
        async def add_memory(content: Annotated[str, "The content to store in memory"]) -> dict:
            """
            add_memory is an MCP tool, this should be called everytime the user informs anything 
            about themselves, their preferences, or anything that has any relevant information 
            which can be useful in the future conversation. This can also be called when the 
            user asks you to remember something.
            
            Returns:
                Result with memory ID and status
            """
            try:
                result = await self.handlers.add_memory(content)
                return result
            except Exception as e:
                logger.error(f"Error adding memory: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.mcp.tool()
        async def search_memory(query: Annotated[str, "The search query. This is the query that the user has asked for."]) -> dict:
            """
            Search for relevant memories based on a query. Use this to find information 
            that was previously stored in memory. This tool should be called anytime user asks 
            for information.  
            

            Example: "What did I tell you about the weather last week?" or 
            "What did I tell you about my friend John?"
                
            Returns:
                List of relevant memories with similarity scores
            """
            try:
                result = await self.handlers.search_memory(query)
                return result
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                return {"status": "error", "message": str(e), "results": []}
        
        @self.mcp.tool()
        async def list_memory(limit: Annotated[int, "The maximum number of memories to return"], offset: Annotated[int, "The number of memories to skip"]) -> dict:
            """
            List all stored memories with pagination.
            
                
            Returns:
                List of memories and total count
            """
            try:
                result = await self.handlers.list_memory(limit, offset)
                return result
            except Exception as e:
                logger.error(f"Error listing memories: {e}")
                return {"status": "error", "message": str(e), "memories": [], "total": 0}
        
        @self.mcp.tool()
        async def delete_all_memory() -> dict:
            """
            Delete all memories from the collection. This will permanently remove 
            all stored memories in the current collection. Use with caution as this 
            operation cannot be undone.
        
                
            Returns:
                Result with deletion status and count of deleted memories
            """
            try:
                result = await self.handlers.delete_all_memory()
                return result
            except Exception as e:
                logger.error(f"Error deleting all memories: {e}")
                return {"status": "error", "message": str(e)}
    
    async def run(self, host: str = "localhost", port: int = 8080):
        """
        Run the MCP server.
        
        Args:
            host: Server host
            port: Server port
        """
        logger.info(f"Starting EzMemory MCP server on {host}:{port}")
        await self.mcp.run_async(transport="http", host=host, port=port)
    
    def get_app(self):
        """Get the FastAPI application."""
        return self.mcp.get_app()
