#!/usr/bin/env python3
"""
Automated paper discovery script for GitHub Actions.

This script fetches recent papers from arXiv, compares them against
tracked keywords using semantic similarity, and adds matching papers
to the database.
"""

import json
import os
import sys
import math
import uuid
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from supabase import create_client
from openai import OpenAI

# Configuration
ARXIV_API_URL = "http://export.arxiv.org/api/query"
EMBEDDING_MODEL = "text-embedding-ada-002"
MATCH_THRESHOLD = 0.75  # Minimum similarity score to consider a match
MAX_PAPERS_PER_RUN = 50  # Maximum papers to fetch per run
DAYS_BACK = 3  # How many days back to search

# Initialize clients
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    response = openai_client.embeddings.create(
        input=text.strip(),
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def fetch_arxiv_papers(categories: List[str], max_results: int = 50) -> List[Dict]:
    """
    Fetch recent papers from arXiv API.

    Args:
        categories: List of arXiv categories to search (e.g., ['cs.AI', 'cs.LG'])
        max_results: Maximum number of results to fetch

    Returns:
        List of paper dictionaries with metadata
    """
    # Build category query
    cat_query = " OR ".join([f"cat:{cat}" for cat in categories])

    params = {
        'search_query': cat_query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    print(f"Fetching papers from arXiv with categories: {categories}")

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    # Parse XML response
    root = ET.fromstring(response.content)

    # Define namespaces
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    papers = []
    for entry in root.findall('atom:entry', ns):
        # Extract arXiv ID from the id URL
        id_url = entry.find('atom:id', ns).text
        arxiv_id = id_url.split('/abs/')[-1]

        # Get PDF URL
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break

        # Get categories
        categories = [cat.get('term') for cat in entry.findall('arxiv:primary_category', ns)]
        categories.extend([cat.get('term') for cat in entry.findall('atom:category', ns)])
        categories = list(set(categories))  # Remove duplicates

        # Get authors
        authors = [author.find('atom:name', ns).text
                   for author in entry.findall('atom:author', ns)]

        # Get published date
        published = entry.find('atom:published', ns).text
        published_date = published.split('T')[0] if published else None

        paper = {
            'arxiv_id': arxiv_id,
            'title': entry.find('atom:title', ns).text.strip().replace('\n', ' '),
            'abstract': entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
            'authors': authors,
            'categories': categories,
            'pdf_url': pdf_url,
            'published_date': published_date
        }
        papers.append(paper)

    print(f"Fetched {len(papers)} papers from arXiv")
    return papers


def get_tracked_keywords() -> List[Dict]:
    """Fetch all active tracked keywords with their embeddings."""
    result = supabase.table('tracked_keywords')\
        .select('id, keyword, embedding')\
        .eq('active', True)\
        .execute()

    return result.data if result.data else []


def get_existing_arxiv_ids() -> set:
    """Get set of arXiv IDs already in the database."""
    result = supabase.table('articles').select('arxiv_id').execute()
    return {article['arxiv_id'] for article in result.data} if result.data else set()


def match_paper_to_keywords(paper: Dict, keywords: List[Dict]) -> List[Dict]:
    """
    Check if a paper matches any tracked keywords.

    Args:
        paper: Paper dictionary with abstract
        keywords: List of keyword dictionaries with embeddings

    Returns:
        List of matching keywords with similarity scores
    """
    if not paper.get('abstract') or not keywords:
        return []

    # Generate embedding for paper abstract
    abstract_embedding = get_embedding(paper['abstract'])

    matches = []
    for kw in keywords:
        if not kw.get('embedding'):
            continue

        embedding = kw['embedding']
        if isinstance(embedding, str):
            embedding = json.loads(embedding)


        similarity = cosine_similarity(abstract_embedding, embedding)

        if similarity >= MATCH_THRESHOLD:
            matches.append({
                'keyword': kw['keyword'],
                'similarity': round(similarity, 4)
            })

    return sorted(matches, key=lambda x: x['similarity'], reverse=True)


def add_paper_to_database(paper: Dict) -> bool:
    """
    Add a paper to the database.

    Args:
        paper: Paper dictionary with metadata

    Returns:
        True if added successfully, False otherwise
    """
    try:
        supabase.table('articles').insert({
            'id': str(uuid.uuid4()),
            'arxiv_id': paper['arxiv_id'],
            'title': paper['title'],
            'authors': paper['authors'],
            'abstract': paper['abstract'],
            'categories': paper['categories'],
            'pdf_url': paper['pdf_url'],
            'published_date': paper['published_date'],
            'processing_status': 'metadata_only',
            'created_at': datetime.utcnow().isoformat()
        }).execute()

        return True
    except Exception as e:
        print(f"Error adding paper {paper['arxiv_id']}: {e}")
        return False


def update_keywords_last_checked():
    """Update the last_checked timestamp for all active keywords."""
    supabase.table('tracked_keywords')\
        .update({'last_checked': datetime.utcnow().isoformat()})\
        .eq('active', True)\
        .execute()


def run_discovery(categories: List[str] = None):
    """
    Main discovery function.

    Args:
        categories: List of arXiv categories to search
    """
    if categories is None:
        # Default AI/ML categories
        categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML']

    print(f"Starting paper discovery at {datetime.utcnow().isoformat()}")
    print(f"Categories: {categories}")
    print(f"Match threshold: {MATCH_THRESHOLD}")

    # Get tracked keywords
    keywords = get_tracked_keywords()
    if not keywords:
        print("No tracked keywords found. Add keywords to enable discovery.")
        return

    print(f"Found {len(keywords)} active keywords")

    # Get existing papers to avoid duplicates
    existing_ids = get_existing_arxiv_ids()
    print(f"Found {len(existing_ids)} existing papers in database")

    # Fetch papers from arXiv
    papers = fetch_arxiv_papers(categories, MAX_PAPERS_PER_RUN)

    # Filter out papers already in database
    new_papers = [p for p in papers if p['arxiv_id'] not in existing_ids]
    print(f"Found {len(new_papers)} new papers to check")

    # Match papers against keywords
    added_count = 0
    for paper in new_papers:
        matches = match_paper_to_keywords(paper, keywords)

        if matches:
            print(f"\nMatched: {paper['title'][:80]}...")
            keyword_strs = [f"{m['keyword']} ({m['similarity']:.2f})" for m in matches[:3]]
            print(f"  Keywords: {', '.join(keyword_strs)}")

            if add_paper_to_database(paper):
                added_count += 1
                print(f"  Added to database!")

    # Update last checked timestamp
    update_keywords_last_checked()

    print(f"\nDiscovery complete!")
    print(f"Papers checked: {len(new_papers)}")
    print(f"Papers added: {added_count}")


if __name__ == "__main__":
    # Parse command line arguments for categories
    import argparse

    parser = argparse.ArgumentParser(description='Discover papers from arXiv matching tracked keywords')
    parser.add_argument('--categories', nargs='+',
                        default=['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML'],
                        help='arXiv categories to search')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Minimum similarity threshold (default: 0.75)')
    parser.add_argument('--max-papers', type=int, default=50,
                        help='Maximum papers to fetch (default: 50)')

    args = parser.parse_args()

    # Update global config
    MATCH_THRESHOLD = args.threshold
    MAX_PAPERS_PER_RUN = args.max_papers

    run_discovery(args.categories)
