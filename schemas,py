
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = "default"

class Citation(BaseModel):
    source: str
    page: str | int

class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    faithful: bool
    reason: str
    language: str
    latency: float

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str

class SearchResponse(BaseModel):
    query: str
    results: list



