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
from .models import Article, ArticleChunk, SearchRequest, SearchResult, KeywordCreate, KeywordResponse

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
    """
    Remove null bytes and control characters from text to ensure database compatibility.
    
    This function is essential for PDF text extraction as PDFs often contain
    binary characters that can cause database insertion errors.
    
    Args:
        text (str): Raw text that may contain control characters
        
    Returns:
        str: Cleaned text safe for database storage
        
    Example:
        >>> clean_text("Hello\x00World\n\n\n")
        "Hello World"
    """
    # Remove null bytes that cause database errors
    text = text.replace('\x00', '')
    
    # Remove other control characters except newlines and tabs
    # Range covers C0 and C1 control characters
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Normalize multiple whitespace characters to single spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks for embedding generation.
    
    Overlapping chunks ensure that concepts spanning chunk boundaries
    are not lost during semantic search. This is crucial for maintaining
    context in research papers.
    
    Args:
        text (str): The text to be chunked
        chunk_size (int): Maximum characters per chunk (default: 1000)
        overlap (int): Characters to overlap between chunks (default: 200)
        
    Returns:
        list[str]: List of text chunks with specified overlap
        
    Example:
        >>> chunks = chunk_text("Long text here...", chunk_size=100, overlap=20)
        >>> len(chunks[0])  # First chunk
        100
        >>> chunks[0][-20:] == chunks[1][:20]  # Overlap check
        True
    """
    # Clean the text first to ensure consistent processing
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return []
        
    chunks = []
    start_position = 0
    text_length = len(cleaned_text)
    
    while start_position < text_length:
        end_position = min(start_position + chunk_size, text_length)
        chunk = cleaned_text[start_position:end_position]
        chunks.append(chunk)
        
        # Move start position forward, accounting for overlap
        start_position += chunk_size - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if start_position <= start_position - (chunk_size - overlap):
            break
    
    return chunks

# Constants for configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-ada-002"
PDF_DOWNLOAD_TIMEOUT = 30

# TODO: should have more optionality here to pick embedding model
# OpenAI client - initialized once for efficiency
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def validate_openai_client() -> bool:
    """
    Validate that OpenAI client is properly configured.
    
    Returns:
        bool: True if client is ready, False otherwise
    """
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        return api_key is not None and len(api_key) > 0
    except Exception:
        return False

def get_embedding_for_text(text: str) -> List[float]:
    """
    Generate embedding for a given text using the configured model.
    
    Args:
        text (str): Text to generate embedding for
        
    Returns:
        List[float]: Embedding vector
        
    Raises:
        Exception: If embedding generation fails
    """
    if not validate_openai_client():
        raise Exception("OpenAI client not properly configured")
        
    response = client.embeddings.create(
        input=text.strip(),
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def validate_article_id(article_id: str) -> None:
    """
    Validate article ID format.
    
    Args:
        article_id (str): Article ID to validate
        
    Raises:
        HTTPException: If article ID is invalid
    """
    if not article_id or not article_id.strip():
        raise HTTPException(status_code=400, detail="Article ID cannot be empty")
    
    # Add more validation as needed (e.g., UUID format check)
    if len(article_id.strip()) < 3:
        raise HTTPException(status_code=400, detail="Article ID too short")

def validate_pagination_params(limit: int, offset: int) -> None:
    """
    Validate pagination parameters.
    
    Args:
        limit (int): Maximum number of items to return
        offset (int): Number of items to skip
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")

# ==========================================
# API ENDPOINTS  
# ==========================================

@app.get("/")
async def root():
    """
    API root endpoint with service information and available endpoints.
    
    Provides an overview of the API and links to key endpoints for
    easy discovery and integration testing.
    
    Returns:
        dict: Service metadata and endpoint directory
    """
    return {
        "service": "Research Paper Tracker API",
        "version": "0.1.0",
        "description": "Track and analyze research papers with semantic search capabilities",
        "endpoints": {
            "health": "/health - API health check",
            "articles": "/articles - List and manage research articles", 
            "search": "/search - Semantic search across paper content",
            "keywords": "/keywords - Manage tracked keywords",
            "stats": "/stats - System statistics"
        },
        "documentation": "/docs",
        "openapi_spec": "/openapi.json"
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check for the API and its dependencies.
    
    This endpoint verifies that the API is running and can connect to
    the Supabase database. Used by monitoring systems and load balancers
    to determine service availability.
    
    Returns:
        dict: Health status information including database connectivity
              and article count for basic functionality verification
              
    Raises:
        HTTPException: 503 (Service Unavailable) if database connection fails
        
    Example Response:
        {
            "status": "healthy",
            "database": "connected", 
            "articles_count": 42
        }
    """
    try:
        # Test database connection with a simple query
        result = supabase.table('articles').select("count").execute()
        
        return {
            "status": "healthy",
            "database": "connected",
            "articles_count": len(result.data),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Database connection failed: {str(e)}"
        )

@app.get("/articles", response_model=List[Article])
async def get_articles(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None
):
    """
    Retrieve articles with pagination and optional status filtering.
    
    This endpoint supports pagination to handle large datasets efficiently
    and allows filtering by processing status to show articles in different states.
    
    Args:
        limit (int): Maximum number of articles to return (default: 20, max recommended: 100)
        offset (int): Number of articles to skip for pagination (default: 0)
        status (Optional[str]): Filter by processing status (e.g., 'metadata_only', 'processing', 'fully_processed')
        
    Returns:
        List[Article]: List of articles matching the criteria
        
    Raises:
        HTTPException: 400 if pagination parameters are invalid
        HTTPException: 500 if database query fails
        
    Example:
        GET /articles?limit=10&offset=20&status=fully_processed
    """
    # Validate input parameters
    validate_pagination_params(limit, offset)
    
    try:
        query = supabase.table('articles').select('*')
        
        # Apply status filter if provided
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
    validate_article_id(article_id)
    
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
    Background task to fully process a research article.
    
    This function performs the complete pipeline for article processing:
    1. Downloads the PDF from the provided URL
    2. Extracts text content from all pages
    3. Cleans and normalizes the text
    4. Splits text into overlapping chunks
    5. Generates embeddings for each chunk using OpenAI
    6. Stores chunks and embeddings in the database
    
    The function handles errors gracefully and updates the article's
    processing status throughout the pipeline.
    
    Args:
        article_id (str): Unique identifier for the article
        article (dict): Article metadata including pdf_url
        
    Side Effects:
        - Updates article processing_status in database
        - Creates article_chunks records with embeddings
        - Logs progress and errors to console
        
    Note:
        This function runs in the background to prevent API timeouts
        for the PDF processing which can take 30+ seconds for large papers.
    """
    # Import PDF processing library with fallback
    try:
        import PyPDF2
    except ImportError:
        import pypdf as PyPDF2  # Handle different package name
    
    try:
        # Mark article as being processed
        supabase.table('articles').update({
            'processing_status': 'processing'
        }).eq('id', article_id).execute()
        
        # Step 1: Download PDF with timeout to prevent hanging
        pdf_url = article['pdf_url']
        print(f"Downloading PDF for article {article_id} from {pdf_url}")
        
        response = requests.get(pdf_url, timeout=PDF_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        
        # Step 2: Extract text from all pages of the PDF
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""
        page_count = len(pdf_reader.pages)
        print(f"Extracting text from {page_count} pages...")
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
                
        if not extracted_text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        print(f"Extracted {len(extracted_text)} characters of text")
        
        # Step 3: Clean and normalize the extracted text
        cleaned_text = clean_text(extracted_text)
        
        # Step 4: Split text into overlapping chunks for embedding
        text_chunks = chunk_text(cleaned_text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
        print(f"Created {len(text_chunks)} text chunks")
        
        # Step 5: Generate embeddings for each chunk using OpenAI
        embedding_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Step 6: Process each chunk and store in database
        for chunk_index, chunk_text in enumerate(text_chunks):
            print(f"Processing chunk {chunk_index + 1}/{len(text_chunks)}")
            
            # Generate embedding for this chunk
            chunk_embedding = get_embedding_for_text(chunk_text)
            
            # Store chunk with embedding in database
            supabase.table('article_chunks').insert({
                'id': str(uuid.uuid4()),
                'article_id': article_id,
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'embedding': chunk_embedding,
                'created_at': datetime.utcnow().isoformat()
            }).execute()
        
        # Step 7: Mark article as fully processed
        supabase.table('articles').update({
            'processing_status': 'fully_processed'
        }).eq('id', article_id).execute()
        
        print(f"Successfully processed article {article_id}: {len(text_chunks)} chunks created")
    
    except Exception as e:
        # Update status to failed
        supabase.table('articles').update({
            'processing_status': 'failed'
        }).eq('id', article_id).execute()
        
        print(f"Failed to process article {article_id}: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search across research paper chunks using vector embeddings.
    
    This endpoint converts the search query to an embedding and finds the most
    semantically similar chunks from processed research papers. The search
    considers conceptual similarity rather than just keyword matching.
    
    Args:
        request (SearchRequest): Contains query string and optional parameters
            - query: The search text to find similar content for
            - limit: Maximum number of results to return (default varies by model)
            
    Returns:
        List[SearchResult]: Ranked list of matching paper chunks with metadata
        
    Raises:
        HTTPException: 500 if embedding generation or database query fails
        
    Example:
        POST /search
        {
            "query": "neural network architecture for natural language processing",
            "limit": 10
        }
        
    Note:
        Requires articles to be fully processed (embeddings generated) to appear in results.
        Search quality depends on the embedding model (currently text-embedding-ada-002).
    """
    try:
        # Convert search query to embedding vector
        query_embedding = get_embedding_for_text(request.query)
        
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
    """
    Get comprehensive system statistics and processing status.
    
    Provides an overview of the system's current state including
    article counts by processing status, total chunks processed,
    and tracked keywords.
    
    Returns:
        dict: System statistics including:
            - Article counts by processing status
            - Total text chunks processed  
            - Tracked keywords count
            - Processing success rate
            
    Raises:
        HTTPException: 500 if database queries fail
    """
    try:
        # Get article statistics
        articles_result = supabase.table('articles').select('processing_status').execute()
        chunks_result = supabase.table('article_chunks').select('id', count='exact').execute()
        keywords_result = supabase.table('tracked_keywords').select('id', count='exact').execute()
        
        articles = articles_result.data
        total_articles = len(articles)
        
        # Count by processing status
        status_counts = {}
        for article in articles:
            status = article['processing_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        processed_count = status_counts.get('fully_processed', 0)
        success_rate = (processed_count / total_articles * 100) if total_articles > 0 else 0
        
        return {
            "total_articles": total_articles,
            "processing_status": status_counts,
            "total_chunks": chunks_result.count if chunks_result.count else 0,
            "tracked_keywords": keywords_result.count if keywords_result.count else 0,
            "processing_success_rate": round(success_rate, 1),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ==========================================
# KEYWORD ENDPOINTS
# ==========================================

@app.get("/keywords", response_model=List[KeywordResponse])
async def get_keywords():
    """
    Retrieve all tracked keywords with their metadata.
    
    Returns a list of all keywords being tracked for research paper monitoring,
    including their activation status and last check timestamps.
    
    Returns:
        List[KeywordResponse]: List of tracked keywords with metadata
        
    Raises:
        HTTPException: 500 if database query fails
    """
    try:
        response = supabase.table('tracked_keywords').select('*').order('created_at', desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch keywords: {str(e)}")


@app.post("/keywords", response_model=KeywordResponse)
async def add_keyword(keyword_data: KeywordCreate):
    """
    Add a new keyword to track with automatic embedding generation.
    
    Creates a new tracked keyword and generates its embedding vector for
    semantic similarity matching against research paper content.
    
    Args:
        keyword_data (KeywordCreate): Contains the keyword/phrase to track
        
    Returns:
        KeywordResponse: The created keyword with metadata
        
    Raises:
        HTTPException: 400 if keyword is empty or already exists
        HTTPException: 500 if embedding generation or database operation fails
        
    Example:
        POST /keywords
        {
            "keyword": "transformer architecture"
        }
    """
    try:
        keyword_text = keyword_data.keyword.strip().lower()
        
        if not keyword_text:
            raise HTTPException(status_code=400, detail="Keyword cannot be empty")
        
        # Check if keyword already exists
        existing = supabase.table('tracked_keywords').select('*').eq('keyword', keyword_text).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Keyword already exists")
        
        # Generate embedding for the keyword using the utility function
        keyword_embedding = get_embedding_for_text(keyword_text)
        
        # Insert into database with generated embedding
        result = supabase.table('tracked_keywords').insert({
            'id': str(uuid.uuid4()),
            'keyword': keyword_text,
            'embedding': keyword_embedding,
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
