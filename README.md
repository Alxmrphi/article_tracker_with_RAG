# 🧠 AI Research Tracking System

A comprehensive platform for tracking, processing, and semantically searching research papers from arXiv and AI conferences. The system automatically monitors papers based on tracked keywords and provides intelligent search capabilities using vector embeddings.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Current Implementation](#current-implementation)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Frontend Features](#frontend-features)
- [Next Steps](#next-steps)
- [Contributing](#contributing)

## 🎯 Overview

This project was initially developed to track **NeuroAI** papers but is designed to be flexible for any research domain. The system combines traditional paper metadata with advanced semantic search capabilities, making it easy to discover relevant research across large paper collections.

### Key Features

- **Automated Paper Tracking**: Monitor arXiv and conferences based on keyword interests
- **Semantic Search**: Find papers by meaning, not just keywords, using vector embeddings
- **Full-Text Processing**: Download and process PDFs to extract complete paper content
- **Interactive Web Interface**: Clean, modern UI for browsing and managing papers
- **Background Processing**: Handle time-intensive PDF processing without blocking the UI
- **Flexible Keyword Management**: Add/remove tracking keywords dynamically

## 🏗️ Architecture

The system follows a modular architecture with clear separation between data processing, API, and presentation layers:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Supabase      │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   Database      │
│                 │    │                 │    │   + Vector DB   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   OpenAI API    │
                    │   (Embeddings)  │
                    └─────────────────┘
```

### Component Details

- **Frontend**: Vanilla HTML/CSS/JS interface for paper management and search
- **FastAPI Backend**: RESTful API handling paper processing, search, and keyword management
- **Supabase Database**: PostgreSQL with vector extensions for storing papers, chunks, and embeddings
- **OpenAI Integration**: Generates embeddings for semantic search capabilities
- **Background Processing**: Async PDF download, text extraction, and embedding generation

## ✅ Current Implementation

### Core Features Implemented

- [x] **Paper Storage & Management**
  - Article metadata storage (title, authors, abstract, etc.)
  - Processing status tracking (metadata_only → processing → fully_processed)
  - PDF URL storage and validation

- [x] **Full-Text Processing Pipeline**
  - PDF download with timeout handling
  - Text extraction using PyPDF2
  - Text cleaning and normalization
  - Chunking with configurable overlap (default: 1000 chars, 200 overlap)
  - OpenAI embedding generation for each chunk

- [x] **Semantic Search**
  - Vector similarity search using Supabase's `match_chunks` function
  - Configurable similarity thresholds and result limits
  - Query embedding generation and matching

- [x] **Keyword Tracking System**
  - Dynamic keyword addition/removal
  - Keyword embedding generation for future matching
  - Active/inactive status management

- [x] **RESTful API**
  - Full CRUD operations for articles and keywords
  - Background processing endpoints
  - Health checks and system statistics
  - Comprehensive error handling

- [x] **Web Interface**
  - Modern, responsive design
  - Paper browsing with status indicators
  - Keyword management interface
  - Processing controls (process papers, view chunks, remove chunks)
  - Real-time statistics dashboard

### Database Schema

```sql
-- Articles table
articles {
  id: uuid (primary key)
  arxiv_id: text
  title: text
  authors: text[]
  abstract: text
  published_date: date
  categories: text[]
  pdf_url: text
  processing_status: text (metadata_only|processing|fully_processed|failed)
  created_at: timestamp
}

-- Article chunks with embeddings
article_chunks {
  id: uuid (primary key)
  article_id: uuid (foreign key)
  chunk_text: text
  chunk_index: integer
  embedding: vector(1536)  -- OpenAI ada-002 dimensions
  created_at: timestamp
}

-- Tracked keywords
tracked_keywords {
  id: uuid (primary key)
  keyword: text
  embedding: vector(1536)
  active: boolean
  created_at: timestamp
  last_checked: timestamp
}
```

## 🚀 Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (for embeddings)
- **Supabase Account** (for database and vector search)
- **PDF Processing Libraries**: PyPDF2 or pypdf

### Environment Variables Required

```bash
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-research-tracking
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn supabase openai PyPDF2 python-dotenv requests
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key
   ```

5. **Set up Supabase database**
   - Create the required tables (see Database Schema)
   - Enable the `vector` extension for embedding storage
   - Create the `match_chunks` RPC function for similarity search

6. **Run the API server**
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```

7. **Open the frontend**
   Open `frontend/index2.html` in your browser

## 📖 Usage

### Adding Papers

Currently papers are added manually to the database. Future versions will include automated arXiv monitoring.

### Processing Papers

1. Papers start in `metadata_only` status
2. Click "Process Paper" to download PDF and extract text
3. System generates embeddings and stores chunks automatically
4. Status updates to `fully_processed` when complete

### Searching Papers

Use the semantic search endpoint to find relevant papers:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks for language understanding", "limit": 5}'
```

### Managing Keywords

Add keywords through the web interface or API:
```bash
curl -X POST "http://localhost:8000/keywords" \
  -H "Content-Type: application/json" \
  -d '{"keyword": "transformer architecture"}'
```

## 🔌 API Documentation

### Main Endpoints

- `GET /` - API overview and endpoint directory
- `GET /health` - Health check with database connectivity
- `GET /articles` - List articles with pagination and filtering
- `POST /articles/{id}/process` - Trigger background PDF processing
- `POST /search` - Semantic search across paper content
- `GET /keywords` - List tracked keywords
- `POST /keywords` - Add new keyword
- `GET /stats` - System statistics and processing status

Full API documentation available at `http://localhost:8000/docs` when running.

## 🎨 Frontend Features

### Dashboard
- Real-time statistics (total papers, processed count, keywords)
- Clean, modern interface with gradient backgrounds
- Responsive design for different screen sizes

### Paper Management
- Card-based layout for easy browsing
- Status indicators with color coding
- One-click processing controls
- Chunk viewing and management

### Keyword Management
- Dynamic keyword addition/removal
- Visual keyword tags with management controls
- Real-time keyword count updates

## 🚧 Next Steps

### High Priority

- [ ] **Automated arXiv Monitoring**
  - Scheduled jobs to check arXiv for new papers
  - Keyword-based filtering during import
  - Automatic processing pipeline

- [ ] **Conference Integration**
  - Support for major AI conferences (NeurIPS, ICML, ICLR, etc.)
  - Conference-specific paper parsing
  - Deadline and event tracking

- [ ] **Enhanced Search Features**
  - Filters by date, author, conference
  - Search result ranking improvements
  - Related paper suggestions

### Medium Priority

- [ ] **User Management & Authentication**
  - Multi-user support with personal collections
  - Sharing and collaboration features
  - Access control and permissions

- [ ] **Advanced Analytics**
  - Research trend analysis
  - Author collaboration networks
  - Topic modeling and clustering

- [ ] **Export & Integration**
  - Export to reference managers (Zotero, Mendeley)
  - BibTeX generation
  - RSS feeds for new papers

### Technical Improvements

- [ ] **Performance Optimization**
  - Caching layer for frequent searches
  - Batch processing for embeddings
  - Database indexing optimization

- [ ] **Monitoring & Observability**
  - Logging and error tracking
  - Processing job monitoring
  - Performance metrics

- [ ] **Deployment & Infrastructure**
  - Docker containerization
  - CI/CD pipeline
  - Production deployment guides

## 🤝 Contributing

This project is in active development. Contributions are welcome for:

- Bug fixes and performance improvements
- New data sources (conferences, journals)
- Enhanced search algorithms
- UI/UX improvements
- Documentation and examples

Please ensure any contributions include appropriate tests and documentation.

## 📄 License

[Add your license information here]

---

**Note**: This project was initially focused on NeuroAI research but is designed to be domain-agnostic. The keyword tracking and semantic search capabilities make it suitable for any research field.
