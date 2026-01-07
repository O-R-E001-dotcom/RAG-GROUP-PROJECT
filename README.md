
## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:

```
OPENAI_API_KEY=our-key-here
```

**Option B: Jump to Production**

```bash

# Build the Rag core
python rag_core.py

# Build authentication
python auh.py 

# Build the schemas for the endpoints
python sche.py

# Start the API
python main.py
```

Visit: http://localhost:8000/docs

---

## Workshop Structure

### Part 1: Test the RAG

**File:** `test.ipynb`

- Load and chunk documents
- Create embeddings with OpenAI
- Store in ChromaDB vector database
- Retrieve relevant chunks
- Generate answers with retrieval tool

### Part 2: Clean Code

**File:** `rag_core.py`

Three simple classes:

1. **DocumentProcessor** - Load & chunk documents
2. **VectorStoreManager** - Embeddings & retrieval
3. **Build_Retrieval_Tool** - Retrieval tool
4. **Build_agent**  - 

### Part 3: Production API

**Files:** `sche.py`, `api.py`, `main.py`

- Build vector index
- FastAPI endpoints
- Auto-generated docs
- Ready to deploy

---

## File Structure

```
Tax-Reform/
├── README.md                        # This file
├── requirements.txt                 # Dependencies
├── .env                            # API keys (create this)
│
├── test.ipynb                      # Part 1: Learn RAG
│
├── rag_core.py                     # Part 2: Clean classes
├── sche.py                  # The schemas for endpoints
├── main.py                          # Part 3: FAST Api
├──auth.py                          # Part 4: Authentication
│
└── documents/                      # Sample documents
    ├── NASS-Journal_Nigeria.pdf
    ├── Nigeria-Tax-Act-2025.pdf
    └── State Government.pdf
```

---

## API Endpoints

### POST /query
Ask a question, get an AI-generated answer

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

### POST /search
Search for relevant documents (no generation)

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Python data types?", "top_k": 3}'
```


---

## Customization

### Change Chunk Size

Edit `rag_core.py`:

```python
doc_processor = DocumentProcessor(
    chunk_size=1000,  # 
    chunk_overlap=100
)
```


### Use GPT-4

Edit `rag_core.py`:

```python
rag_generator = RAGGenerator(
    vectorstore_manager,
    openai_model="gpt-4"  
)
```

### "Port 8000 already in use"

Use a different port:
```bash
uvicorn api:app --port 8001
```

---

## Dependencies

- **LangChain** - RAG orchestration
- **ChromaDB** - Vector database
- **OpenAI** - Embeddings & LLM
- **FastAPI** - API framework
- **uvicorn**
- **python-dotenv**
- **langgraph**
- **langdetect**
- **pydantic**
- **python-jose**
- **passlib[bcrypt]**






