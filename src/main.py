from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel
from datetime import datetime

import re
import os
import uuid
import requests
from io import BytesIO

from .database import supabase
from .models import Article, ArticleChunk, SearchRequest, SearchResult

app = FastAPI(
    title="Research Paper Tracker API",
    description="Track and analyse research papers",
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

def clean_text(text: str) -> str:
    """Remove null bytes and control characters from text"""
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks"""
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# TODO: should have more optionality here to pick embedding model
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

@app.post("/articles/{article_id}/process")
async def process_article(article_id: str, background_tasks: BackgroundTasks):
    """
    Trigger full processing of a paper (download PDF, extract text, generate embeddings)
    Runs in background to avoid timeout
    """
    try:
        # First check if article exists and needs processing
        result = supabase.table('articles').select('*').eq('id', article_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Article not found")
        
        article = result.data[0]
        
        if article['processing_status'] == 'fully_processed':
            raise HTTPException(status_code=400, detail="Article already fully processed")
        
        # Add processing task to background
        background_tasks.add_task(process_article_background, article_id, article)
        
        return {
            "message": "Processing started",
            "article_id": article_id,
            "status": "processing"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

async def process_article_background(article_id: str, article: dict):
    """
    Background task to fully process an article
    This is the function that does the actual work
    """
    try:
        import PyPDF2
    except ImportError:
        import pypdf as PyPDF2  # Handle different package name
    
    try:
        # Update status to processing
        supabase.table('articles').update({
            'processing_status': 'processing'
        }).eq('id', article_id).execute()
        
        # Step 1: Download PDF
        pdf_url = article['pdf_url']
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Step 2: Extract text from PDF
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        if not full_text.strip():
            raise Exception("No text extracted from PDF")
        
        # Step 3: Clean text
        full_text = clean_text(full_text)
        
        # Step 4: Chunk text (1000 chars with 200 char overlap)
        chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
        
        # Step 5: Generate embeddings for each chunk
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Step 6: Insert chunks into database
        for idx, chunk in enumerate(chunks):
            # Generate embedding
            embedding_response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = embedding_response.data[0].embedding
            
            # Insert chunk
            supabase.table('article_chunks').insert({
                'id': str(uuid.uuid4()),
                'article_id': article_id,
                'chunk_text': chunk,
                'chunk_index': idx,
                'embedding': embedding,
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        
        # Step 7: Update article status to fully_processed
        supabase.table('articles').update({
            'processing_status': 'fully_processed'
        }).eq('id', article_id).execute()
        
        print(f"Successfully processed article {article_id}: {len(chunks)} chunks created")
    
    except Exception as e:
        # Update status to failed
        supabase.table('articles').update({
            'processing_status': 'failed'
        }).eq('id', article_id).execute()
        
        print(f"Failed to process article {article_id}: {str(e)}")

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
    
# ==========================================
# KEYWORD ENDPOINTS
# ==========================================

@app.get("/keywords")
async def get_keywords():
    """Get all tracked keywords"""
    try:
        response = supabase.table('tracked_keywords').select('*').order('created_at', desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch keywords: {str(e)}")


@app.post("/keywords")
async def add_keyword(keyword_data: dict):
    """Add a new keyword to track (generates embedding automatically)"""
    try:
        keyword = keyword_data.get('keyword', '').strip().lower()
        if not keyword:
            raise HTTPException(status_code=400, detail="Keyword cannot be empty")
        
        # Generate embedding
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.embeddings.create(
            input=keyword,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        
        # Insert into database
        result = supabase.table('tracked_keywords').insert({
            'id': str(uuid.uuid4()),
            'keyword': keyword,
            'embedding': embedding,
            'active': True,
            'created_at': datetime.utcnow().isoformat(),
            'last_checked': None
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to insert keyword")
        
        return result.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add keyword: {str(e)}")


@app.delete("/keywords/{keyword_id}")
async def delete_keyword(keyword_id: str):
    """Delete a tracked keyword"""
    try:
        result = supabase.table('tracked_keywords').delete().eq('id', keyword_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        return {"message": "Keyword deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete keyword: {str(e)}")


# ==========================================
# PAPER PROCESSING ENDPOINT
# ==========================================

@app.delete("/articles/{article_id}/chunks")
async def delete_article_chunks(article_id: str):
    """Delete all chunks for an article and reset status to metadata_only"""
    try:
        # Delete all chunks for this article
        supabase.table('article_chunks').delete().eq('article_id', article_id).execute()
        
        # Reset article status to metadata_only
        result = supabase.table('articles').update({
            'processing_status': 'metadata_only'
        }).eq('id', article_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return {
            "message": "Chunks deleted and article status reset",
            "article_id": article_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {str(e)}")

@app.post("/articles/{article_id}/process")
async def process_article(article_id: str, background_tasks: BackgroundTasks):
    """Trigger full processing of a paper (download PDF, extract text, generate embeddings)"""
    try:
        # Check if article exists and needs processing
        result = supabase.table('articles').select('*').eq('id', article_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Article not found")
        
        article = result.data[0]
        
        if article['processing_status'] == 'fully_processed':
            raise HTTPException(status_code=400, detail="Article already fully processed")
        
        # Add processing task to background
        background_tasks.add_task(process_article_background, article_id, article)
        
        return {
            "message": "Processing started",
            "article_id": article_id,
            "status": "processing"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")
    """Delete a tracked keyword"""
    try:
        result = supabase.table('tracked_keywords').delete().eq('id', keyword_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Keyword not found")
        
        return {"message": "Keyword deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete keyword: {str(e)}")



    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence or word boundary
        if end < text_length:
            # Look for sentence boundary (. ! ?)
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
            else:
                # If no sentence boundary, look for word boundary
                for i in range(end, max(start + chunk_size - 50, start), -1):
                    if text[i].isspace():
                        end = i
                        break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap
        
        # Ensure we make progress
        if start <= chunks[-1] if chunks else 0:
            start = end
    
    return chunks
