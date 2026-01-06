
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError

from schemas import (
    ChatRequest,
    ChatResponse,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
)

from rag_core import build_rag_system
from auth import (
    users_db,
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
)

app = FastAPI(title="Nigerian Tax RAG API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build agent once at startup
print("🚀 Initializing RAG system...")
chat_agent = build_rag_system("documents")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Nigerian Tax Reform RAG API",
        "endpoints": {
            "register": "Register a new user (POST)",
            "login": "Login and get access token (POST)",
            "/chat": "Chat with the AI assistant (POST)"
        }
    }

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


def get_current_user(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        payload = decode_token(token)
        return payload["sub"]
    except JWTError:
        raise HTTPException(401, "Invalid token")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user=Depends(get_current_user)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        # IMPORTANT: agent returns (answer, citations)
        answer, citations = chat_agent(
            user_input=req.message,
            thread_id=user
        )

        return {
            "answer": answer,
            "citations": citations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


