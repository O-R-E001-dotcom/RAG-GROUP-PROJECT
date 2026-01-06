
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from jose import jwt, JWTError
from passlib.context import CryptContext
import bcrypt
import hashlib
import base64
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("EXPIRY"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory users (replace with DB later, we can use MySQL or Supabase)
users_db = {}

def hash_password(password: str):
    # Step 1: hash password to 32-byte digest
    password_hash = hashlib.sha256(password.encode("utf-8")).digest()
    # Step 2: hash with bcrypt
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_hash, salt)
     # Encode to base64 string for safe storage
    return base64.b64encode(hashed).decode("utf-8")
    
def verify_password(password: str, hashed: str) -> bool:
    # SHA256
    password_hash = hashlib.sha256(password.encode("utf-8")).digest()
    # Decode stored hash from base64
    stored_hash = base64.b64decode(hashed.encode("utf-8"))
    return bcrypt.checkpw(password_hash, stored_hash)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

# Utility: Token dependency

def get_current_user(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        payload = decode_token(token)
        return payload["sub"]
    except JWTError:
        raise HTTPException(401, "Invalid token")
