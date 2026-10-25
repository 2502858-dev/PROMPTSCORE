from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24   # 24 hours

security = HTTPBearer()


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency — use as: current_user: dict = Depends(get_current_user)
    Returns the decoded token payload (user_id/doctor_id, email, role)."""
    payload = verify_token(credentials.credentials)
    return payload


def require_role(*allowed_roles: str):
    """Dependency factory — use as: Depends(require_role('doctor'))
    Rejects the request with 403 if the token's role isn't in allowed_roles."""
    def checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient permissions")
        return current_user
    return checker