
# from fastapi import FastAPI, HTTPException, Depends, Header
# from fastapi.middleware.cors import CORSMiddleware
# from jose import JWTError
# from sche import ChatRequest, ChatResponse, RegisterRequest, LoginRequest, TokenResponse
# from rag_core import DocumentProcessor, VectorStoreManager, build_retrieval_tool, build_agent as build_rag_system
# from auth import (
#     users_db,
#     hash_password,
#     verify_password,
#     create_access_token,
#     decode_token,
# )

# app = FastAPI(title="Nigerian Tax RAG API")

# # Allow frontend access
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict later
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Build agent once at startup
# print("🚀 Initializing RAG system...")
# chat_agent = build_rag_system("documents")

# @app.get("/")
# async def root():
#     """Root endpoint."""
#     return {
#         "message": "Nigerian Tax Reform RAG API",
#         "endpoints": {
#             "register": "Register a new user (POST)",
#             "login": "Login and get access token (POST)",
#             "/chat": "Chat with the AI assistant (POST)"
#         }
#     }

# @app.post("/register", response_model=TokenResponse)
# def register(req: RegisterRequest):
#     if req.email in users_db:
#         raise HTTPException(400, "User exists")

#     users_db[req.email] = hash_password(req.password)
#     token = create_access_token({"sub": req.email})
#     return {"access_token": token}


# @app.post("/login", response_model=TokenResponse)
# def login(req: LoginRequest):
#     hashed = users_db.get(req.email)
#     if not hashed or not verify_password(req.password, hashed):
#         raise HTTPException(401, "Invalid credentials")

#     token = create_access_token({"sub": req.email})
#     return {"access_token": token}


# def get_current_user(authorization: str = Header(...)):
#     try:
#         token = authorization.replace("Bearer ", "")
#         payload = decode_token(token)
#         return payload["sub"]
#     except JWTError:
#         raise HTTPException(401, "Invalid token")

# @app.post("/chat", response_model=ChatResponse)
# async def chat(req: ChatRequest, user=Depends(get_current_user)):
#     if not req.message.strip():
#         raise HTTPException(status_code=400, detail="Empty message")

#     try:
#         response = chat_agent(user_input=req.message, thread_id=user)

#         return {
#             "question": req.message,
#             "answer": response["answer"],
#             "citations": response["sources"],
#             "faithful": response["faithful"],
#             "reason": response.get("faithfulness_reason", ""),
#             "language": response["language"],
#             "latency": 0.0  # Placeholder for latency measurement
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# main.py
import os
import traceback
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError
from sche import ChatRequest, ChatResponse, RegisterRequest, LoginRequest, TokenResponse
from rag_core import (
    # DocumentProcessor,
    # VectorStoreManager,
    # build_retrieval_tool,
    build_agent as build_rag_system
)
from auth import (
    users_db,
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    get_current_user
)

# =========================
# FastAPI app setup
# =========================
app = FastAPI(title="Nigerian Tax RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Initialize RAG system at startup
# =========================
print("🚀 Initializing RAG system...")

# try:
#     # 1️⃣ Load documents
#     doc_processor = DocumentProcessor(folder="folder")
#     docs = doc_processor.load_documents()
#     chunks = doc_processor.chunk_documents(docs)

#     # 2️⃣ Create vector store
#     vector_manager = VectorStoreManager()
#     vector_manager.create_vectorstore(chunks)
#     vectorestore = vector_manager.load_vectorstore()

#     # 3️⃣ Build RAG agent
#     chat_agent_helpers = build_rag_system(vectorstore=vectorestore)
#     chat_agent = chat_agent_helpers["query_agent"]

#     print("✅ RAG system ready!")

# except Exception as e:
#     print("❌ Failed to initialize RAG system:")
#     print(str(e))
#     traceback.print_exc()
#     chat_agent = None




# =========================
# Root endpoint
# =========================
@app.get("/")
async def root():
    return {
        "message": "Nigerian Tax Reform RAG API",
        "endpoints": {
            "register": "Register a new user (POST)",
            "login": "Login and get access token (POST)",
            "/chat": "Chat with the AI assistant (POST)"
        }
    }

# =========================
# User auth endpoints
# =========================
@app.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    if req.email in users_db:
        raise HTTPException(400, "User exists")

    users_db[req.email] = hash_password(req.password)
    token = create_access_token({"sub": req.email})
    return {"access_token": token}


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    hashed = users_db.get(req.email)
    if not hashed or not verify_password(req.password, hashed):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"sub": req.email})
    return {"access_token": token}

# =========================
# Chat endpoint
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    if chat_agent is None:
        raise HTTPException(500, "RAG agent is not initialized")

    try:
        response = chat_agent(user_input=req.message, thread_id=user)

        return {
            "question": req.message,
            "answer": response["answer"],
            "citations": response["sources"],
            "faithful": response["faithful"],
            "reason": response.get("faithfulness_reason", ""),
            "language": response["language"],
            "latency": 0.0  # optional: measure if needed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Run server (if executed directly)
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
