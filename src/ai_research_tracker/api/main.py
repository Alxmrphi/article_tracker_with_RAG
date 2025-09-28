from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
from openai import OpenAI

from .database import supabase
from .models import Article, ArticleChunk, SearchRequest, SearchResult

app = FastAPI(
    title="SUB-AI Research Tracker API",
    description="Track and analyse NeuroAI research papers",
    version="0.1.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "SUB-AI Research Tracker API",
        "version": "0.1.0",
        "endpoints": {
            "articles": "/articles",
            "search": "/search",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Check API and database health"""
    try:
        # Test database connection
        result = supabase.table('articles').select("count").execute()
        return {
            "status": "healthy",
            "database": "connected",
            "articles_count": len(result.data)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

@app.get("/articles", response_model=List[Article])
async def get_articles(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None
):
    """Get list of articles with pagination"""
    try:
        query = supabase.table('articles').select('*')
        
        if status:
            query = query.eq('processing_status', status)
        
        result = query.order('created_at', desc=True)\
                     .range(offset, offset + limit - 1)\
                     .execute()
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/articles/{article_id}", response_model=Article)
async def get_article(article_id: str):
    """Get a specific article by ID"""
    try:
        result = supabase.table('articles')\
            .select('*')\
            .eq('id', article_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/articles/{article_id}/chunks", response_model=List[ArticleChunk])
async def get_article_chunks(article_id: str):
    """Get all chunks for a specific article"""
    try:
        result = supabase.table('article_chunks')\
            .select('*')\
            .eq('article_id', article_id)\
            .order('chunk_index')\
            .execute()
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def semantic_search(request: SearchRequest):
    """Search papers semantically using embeddings"""
    try:
        # Generate embedding for search query
        response = client.embeddings.create(
            input=request.query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        
        # Search using vector similarity
        # Note: This uses Supabase's vector search
        result = supabase.rpc(
            'match_chunks',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,
                'match_count': request.limit
            }
        ).execute()
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        articles_result = supabase.table('articles').select('processing_status').execute()
        chunks_result = supabase.table('article_chunks').select('id', count='exact').execute()
        
        articles = articles_result.data
        total_articles = len(articles)
        processed = len([a for a in articles if a['processing_status'] == 'fully_processed'])
        metadata_only = len([a for a in articles if a['processing_status'] == 'metadata_only'])
        
        return {
            "total_articles": total_articles,
            "fully_processed": processed,
            "metadata_only": metadata_only,
            "total_chunks": chunks_result.count if chunks_result.count else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))