from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date

class Article(BaseModel):
    id: str
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    published_date: Optional[date] = None
    categories: List[str]
    pdf_url: Optional[str] = None
    processing_status: str
    created_at: datetime # Check if needed

class ArticleChunk(BaseModel):
    id: str
    article_id: str
    chunk_text: str
    chunk_index: int
    created_at: datetime

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class SearchResult(BaseModel):
    article: Article
    similarity_score: float
    matching_chunk: str

class KeywordCreate(BaseModel):
    keyword: str # TODO: phrase, not keyword

class KeywordResponse(BaseModel): # Same here
    id: str
    keyword: str
    active: bool
    created_at: datetime
    last_checked: Optional[datetime]
