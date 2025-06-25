import logging
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import jwt

from ..mlops.database import DatabaseManager, User


class APIAuth:
    """Handles API authentication and authorization."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_urlsafe(64)
        self.algorithm = "HS256"
        self.token_expire_minutes = 60 * 24  # 24 hours
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.db_manager = DatabaseManager()
        self.logger = logging.getLogger(__name__)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = self.db_manager.get_user(username)
        if not user or not self.verify_password(password, user.password_hash):
            return None
        return user

    def get_current_user(self, token: str = Depends(HTTPBearer())) -> User:
        try:
            payload = jwt.decode(token.credentials, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid authentication credentials")
            user = self.db_manager.get_user(username)
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            return user
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    def require_role(self, required_role: str):
        def check_role(user: User = Depends(self.get_current_user)):
            if required_role not in user.roles:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Role '{required_role}' required"
                )
            return user
        return check_role


auth_instance = APIAuth()
require_auth = auth_instance.get_current_user
require_admin = auth_instance.require_role("admin")