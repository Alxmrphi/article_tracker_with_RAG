# AI Research Tracker - Project Overview

## Project Purpose

An automated system for tracking, processing, and analyzing AI research papers from arXiv and academic conferences. The system uses vector embeddings for semantic search and provides LLM-powered analysis through Retrieval Augmented Generation (RAG).

**Primary Learning Objectives:**
- Understand modern backend infrastructure and data pipelines
- Learn vector databases and semantic search
- Gain hands-on experience with API development
- Build production-ready automation workflows

---

## Full Intended Functionality

### Core Features
1. **Automated Paper Discovery**
   - Daily monitoring of arXiv for new papers
   - Keyword-based matching using semantic similarity
   - Automatic addition of relevant papers to tracking database

2. **Paper Processing Pipeline**
   - PDF download and text extraction
   - Intelligent text chunking with overlap
   - OpenAI embedding generation for each chunk
   - Vector storage for semantic search

3. **Semantic Search**
   - Query papers using natural language
   - Find relevant content across all processed papers
   - Vector similarity matching (cosine similarity)

4. **RAG Query System**
   - Ask questions about your research collection
   - LLM generates answers using retrieved paper chunks
   - Source citations from relevant papers

5. **Keyword Management**
   - Add/remove tracked keywords
   - Keywords stored with embeddings for matching
   - Track when keywords were last checked

6. **Web Interface**
   - View all tracked papers
   - Process papers on-demand
   - Manage keywords
   - View paper chunks
   - Search and query interface

### Future Enhancements
- Conference paper integration (NeurIPS, ICML, ICLR, CVPR)
- AWS migration for learning cloud infrastructure
- Paper recommendation system
- Trend analysis dashboard
- Citation tracking
- Author following

---

## Tech Stack

**Backend:**
- Python 3.x (managed with `uv`)
- FastAPI (REST API framework)
- Supabase (PostgreSQL + pg_vector extension)
- OpenAI API (text-embedding-ada-002 for embeddings, gpt-4o-mini for RAG)

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- Fetch API for backend communication

**Automation:**
- GitHub Actions (scheduled workflows for paper discovery)

**Deployment:**
- Currently: Local development
- Planned: Vercel for API, potential AWS migration

---

## Database Schema

### Tables

#### 1. `articles`
Stores paper metadata from arXiv/conferences.
- Paper identifiers (id, arxiv_id)
- Metadata (title, authors, abstract, published_date, categories)
- PDF URL
- Processing status tracking (metadata_only, processing, fully_processed, failed)

#### 2. `article_chunks`
Stores processed paper text with embeddings.
- Links to parent article (foreign key)
- Chunk text (~1000 characters)
- Chunk index (ordering)
- Vector embedding (1536 dimensions)

#### 3. `tracked_keywords`
User-defined keywords for automated discovery.
- Keyword text
- Keyword embedding
- Active status
- Last checked timestamp

---

## What Has Been Completed ✅

### Infrastructure Setup
- [x] Project initialized with `uv` package manager
- [x] GitHub repository configured
- [x] Development environment with Jupyter notebooks
- [x] Environment variables configured (.env)
- [x] All dependencies installed

### Database
- [x] Supabase PostgreSQL database created
- [x] pg_vector extension enabled
- [x] All 3 tables created with proper schema
- [x] Foreign key relationships established
- [x] Vector similarity search function (`match_chunks`) created
- [x] Appropriate indexes (B-tree, GIN, IVFFlat)

### arXiv Integration
- [x] arXiv API exploration and understanding
- [x] Query construction with categories and operators
- [x] Helper functions for fetching papers
- [x] Date-based filtering
- [x] Automated paper discovery script

### Paper Processing Pipeline
- [x] PDF download functionality
- [x] Text extraction (PyPDF2)
- [x] Text cleaning (removes control characters, normalizes whitespace)
- [x] Smart chunking with overlap (1000 chars, 200 overlap)
- [x] OpenAI embedding generation
- [x] Chunk storage with embeddings
- [x] Processing status tracking (including failed state)

### FastAPI Backend
- [x] API application structure (`main.py`, `database.py`, `models.py`)
- [x] CORS middleware configured
- [x] Health check endpoint
- [x] Article listing endpoint (with pagination/filtering)
- [x] Article detail endpoint
- [x] Article chunks endpoint
- [x] Semantic search endpoint
- [x] Stats endpoint
- [x] Keyword CRUD endpoints (GET, POST, DELETE)
- [x] Paper processing endpoint (background task)
- [x] Chunk deletion endpoint (unprocess papers)
- [x] Article reset endpoint (retry failed papers)
- [x] RAG query endpoint (`/query`) - LLM-powered Q&A
- [x] Keyword matching endpoints (`/keywords/match-paper`, `/keywords/discover`)

### RAG Query System
- [x] RAG endpoint in FastAPI (`POST /query`)
- [x] Retrieves relevant chunks via vector similarity
- [x] Sends chunks + question to OpenAI GPT-4o-mini
- [x] Returns answer with source citations
- [x] Context window management (12k char limit)
- [x] Query interface in frontend

### Keyword Matching System
- [x] Cosine similarity comparison function
- [x] Compare papers against tracked keywords
- [x] Configurable similarity threshold (default: 0.75)
- [x] Match paper endpoint (`/keywords/match-paper/{article_id}`)
- [x] Discover papers endpoint (`/keywords/discover`)
- [x] Updates keyword last_checked timestamp

### GitHub Actions Automation
- [x] Daily paper discovery workflow (runs at 6 AM UTC)
- [x] Broad collection workflow (runs every 3 days)
- [x] Discovery script with arXiv API integration
- [x] Configurable categories and thresholds
- [x] Manual workflow trigger support

### Frontend Interface
- [x] Two complete UI designs (table and card-based)
- [x] Paper listing with status indicators
- [x] Process/reprocess paper functionality
- [x] View chunks functionality
- [x] Remove chunks functionality
- [x] Reset failed papers functionality
- [x] Keyword management interface
- [x] Add/remove keywords
- [x] Real-time stats display
- [x] Error handling and loading states
- [x] RAG query interface ("Ask Your Papers")
- [x] Query results with source citations

### Development Tools
- [x] Jupyter notebooks for exploration
- [x] API documentation (FastAPI auto-docs)

---

## What Remains To Do ⏳

### Medium Priority - Polish & Features

#### 1. Frontend Enhancements
- [ ] Implement pagination for large paper lists
- [ ] Add search history
- [ ] Add loading animations
- [ ] Display keyword match results in UI

#### 2. Backend Improvements
- [ ] Add rate limiting
- [ ] Add logging system
- [ ] Create processing queue for multiple papers
- [ ] Optimize vector search performance
- [ ] Add caching layer

#### 3. Testing & Documentation
- [ ] Write unit tests for processing pipeline
- [ ] Create API integration tests
- [ ] Document API endpoints
- [ ] Create user guide
- [ ] Add code comments
- [ ] Create deployment guide

### Low Priority - Advanced Features

#### 4. Conference Integration
- [ ] Research NeurIPS website structure
- [ ] Implement web scraping
- [ ] Handle different paper formats
- [ ] Expand to ICML, ICLR, CVPR

#### 5. AWS Migration (Learning Focus)
- [ ] Design AWS architecture
- [ ] Set up S3 for PDF storage
- [ ] Create Lambda functions for processing
- [ ] Configure EventBridge for scheduling
- [ ] Set up CloudWatch monitoring
- [ ] Compare costs with current stack

#### 6. Advanced Analytics
- [ ] Suggested keywords based on processed papers
- [ ] Paper recommendation system
- [ ] Trend analysis over time
- [ ] Citation network visualization
- [ ] Author following feature
- [ ] Topic clustering

---

## Current Project Status

**Phase:** Core functionality complete including RAG and automation

**Working:**
- ✅ Manual paper tracking and processing
- ✅ Keyword management (CRUD operations)
- ✅ Basic semantic search
- ✅ Web interface for all operations
- ✅ Failed paper recovery (reset functionality)
- ✅ RAG query system (ask questions, get LLM answers with sources)
- ✅ Keyword matching (compare papers against tracked keywords)
- ✅ Automated paper discovery (GitHub Actions)

**Next Session Focus:**
1. Add pagination to frontend for large paper lists
2. Implement logging system for better debugging
3. Write unit tests for critical functions

**Known Issues:**
- Earlier processed papers have formatting issues (newlines) - can be reprocessed
- Discovery workflow requires GitHub Secrets to be configured

---

## File Structure

```
ai-research-tracker/
├── .github/
│   └── workflows/
│       ├── daily-discovery.yml    # Daily paper discovery (6 AM UTC)
│       └── broad-collection.yml   # Broad collection (every 3 days)
├── src/
│   ├── __init__.py               # Package init
│   ├── main.py                   # FastAPI application
│   ├── database.py               # Supabase connection
│   └── models.py                 # Pydantic models
├── frontend/
│   ├── index.html                # Table-based UI
│   └── index2.html               # Card-based UI (with RAG query)
├── scripts/
│   └── discover_papers.py        # Automated paper discovery script
├── notebooks/
│   └── arxiv_explorer.ipynb      # arXiv API exploration
├── pdf_extraction.ipynb          # PDF processing experiments
├── supabase_test.ipynb           # Database testing
├── .env                          # Environment variables (gitignored)
├── .gitignore
├── pyproject.toml                # uv configuration
├── CLAUDE.md                     # This file
└── README.md
```

---

## Environment Variables Required

```env
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGc...
OPENAI_API_KEY=sk-proj-...
```

**For GitHub Actions (add as repository secrets):**
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `OPENAI_API_KEY`

---

## Running the Project

**Start API:**
```bash
uvicorn src.main:app --reload
```

**Access Frontend:**
- Open `frontend/index.html` or `frontend/index2.html` in browser
- API docs available at `http://localhost:8000/docs`

**Run Paper Discovery Manually:**
```bash
uv run python scripts/discover_papers.py --categories cs.AI cs.LG --threshold 0.75
```

**Development:**
- Jupyter notebooks for experimentation
- Test API endpoints via FastAPI docs interface

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and endpoint directory |
| GET | `/health` | Health check with database status |
| GET | `/articles` | List articles (with pagination) |
| GET | `/articles/{id}` | Get single article |
| GET | `/articles/{id}/chunks` | Get article chunks |
| POST | `/articles/{id}/process` | Process article (background) |
| POST | `/articles/{id}/reset` | Reset failed article |
| DELETE | `/articles/{id}/chunks` | Delete article chunks |
| POST | `/search` | Semantic search |
| POST | `/query` | RAG query (LLM-powered Q&A) |
| GET | `/keywords` | List tracked keywords |
| POST | `/keywords` | Add keyword |
| DELETE | `/keywords/{id}` | Delete keyword |
| POST | `/keywords/match-paper/{id}` | Match paper against keywords |
| GET | `/keywords/discover` | Discover matching papers |
| GET | `/stats` | System statistics |

---

## Key Learning Concepts Covered

### Database & SQL
- PostgreSQL schema design
- Foreign keys and relationships
- Array and vector data types
- Indexes (B-tree, GIN, IVFFlat)
- SQL functions

### Vector Embeddings
- OpenAI embeddings API (v1.0+)
- Cosine similarity search
- pg_vector extension
- Semantic search principles

### API Development
- FastAPI framework
- RESTful design
- Background tasks
- CORS configuration
- Pydantic validation

### Text Processing
- PDF extraction
- Text cleaning and normalization
- Chunking strategies
- Character encoding handling

### Frontend Development
- Vanilla JavaScript
- Fetch API
- Dynamic DOM manipulation
- Multiple UI design approaches

### LLM Integration
- OpenAI Chat Completions API
- Prompt engineering for RAG
- Context window management
- Temperature tuning for factual responses

### CI/CD & Automation
- GitHub Actions workflows
- Scheduled tasks (cron)
- Environment secrets management
- Workflow inputs for configurability

---

## Notes for Future Sessions

**Development Philosophy:**
- Prioritize understanding over speed
- Build solid foundations before adding complexity
- Focus on patterns applicable beyond this project
- Hands-on experimentation with real data

**User Preferences:**
- Detailed explanations of concepts
- Understanding the "why" behind decisions
- Working on macOS
- Using VSCode as primary IDE
- Comfortable with Python, learning SQL

**GitHub Repository:**
https://github.com/Alxmrphi/neuroai_papers

---

## Changelog

### 2026-01-11

**RAG Query System Implementation**
- Added `POST /query` endpoint for LLM-powered question answering
- Retrieves relevant chunks using vector similarity search
- Sends context + question to OpenAI GPT-4o-mini
- Returns structured response with answer and source citations
- Added Pydantic models: `RAGQueryRequest`, `RAGQueryResponse`, `RAGSourceChunk`
- Frontend: Added "Ask Your Papers" section with query input and results display
- Configurable max_chunks (default: 5) and similarity_threshold (default: 0.7)
- Context window management with 12k character limit

**Keyword Matching System Implementation**
- Added `cosine_similarity()` function for vector comparison
- Added `POST /keywords/match-paper/{article_id}` endpoint
  - Compares paper abstract against all tracked keywords
  - Returns matching keywords sorted by similarity score
- Added `GET /keywords/discover` endpoint
  - Finds papers matching any tracked keyword
  - Returns papers with their matching keywords and best scores
  - Updates keyword `last_checked` timestamps
- Added Pydantic models: `KeywordMatchResult`, `PaperMatchResult`

**GitHub Actions Automation**
- Created `daily-discovery.yml` workflow
  - Runs daily at 6 AM UTC
  - Searches AI/ML categories (cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, stat.ML)
  - Uses 0.75 similarity threshold
- Created `broad-collection.yml` workflow
  - Runs every 3 days at 3 AM UTC
  - Includes additional categories (q-bio.NC, cs.RO)
  - Supports manual trigger with configurable parameters
- Created `scripts/discover_papers.py`
  - Fetches papers from arXiv API
  - Matches against tracked keywords using embeddings
  - Automatically adds matching papers to database
  - Command-line arguments for categories, threshold, max papers

**Files Changed:**
- `src/main.py` - Added RAG and keyword matching endpoints
- `src/models.py` - Added new Pydantic models
- `frontend/index2.html` - Added RAG query interface
- `scripts/discover_papers.py` - New file
- `.github/workflows/daily-discovery.yml` - New file
- `.github/workflows/broad-collection.yml` - New file
- `CLAUDE.md` - Updated documentation
