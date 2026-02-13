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
    article_id: str
    chunk_text: str
    similarity: float

class KeywordCreate(BaseModel):
    keyword: str # TODO: phrase, not keyword

class KeywordResponse(BaseModel): # Same here
    id: str
    keyword: str
    active: bool
    created_at: datetime
    last_checked: Optional[datetime]


# RAG Query Models
class RAGQueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str
    max_chunks: int = 5  # Maximum number of chunks to retrieve for context
    similarity_threshold: float = 0.7  # Minimum similarity score for chunks


class RAGSourceChunk(BaseModel):
    """A source chunk used in RAG response"""
    article_id: str
    article_title: str
    chunk_text: str
    similarity_score: float


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries"""
    question: str
    answer: str
    sources: List[RAGSourceChunk]
    chunks_used: int


# Keyword Matching Models
class KeywordMatchResult(BaseModel):
    """Result of matching a paper against tracked keywords"""
    keyword_id: str
    keyword: str
    similarity_score: float


class PaperMatchResult(BaseModel):
    """Paper with its keyword match scores"""
    arxiv_id: str
    title: str
    abstract: str
    matching_keywords: List[KeywordMatchResult]
    best_match_score: float
