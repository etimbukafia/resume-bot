from pydantic import BaseModel, Field 
from typing import List, Any, Optional

class QueryRequest(BaseModel):
    query: str

#class SearchResponse(BaseModel):
    #listings: List[Listing]