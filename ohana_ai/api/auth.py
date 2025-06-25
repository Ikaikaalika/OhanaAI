"""
Authentication and authorization for the OhanaAI API.
"""

import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class APIAuth:
    """Handles API authentication and authorization."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize API authentication.
        
        Args:
            secret_key: Secret key for JWT signing (generates random if not provided)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(64)
        self.algorithm = "HS256"
        self.token_expire_hours = 24
        self.logger = logging.getLogger(__name__)
        
        # Simple user store (in production, use proper database)
        self.users = {
            "admin": {
                "password_hash": self._hash_password("admin123"),  # Change default password!
                "roles": ["admin", "user"],
                "active": True
            },
            "user": {
                "password_hash": self._hash_password("user123"),
                "roles": ["user"],
                "active": True
            }
        }
        
        # API key store
        self.api_keys = {
            "ohana_api_key_change_me": {
                "user": "api_user",
                "roles": ["user"],
                "active": True,
                "created_at": datetime.now()
            }
        }

    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        return self._hash_password(password) == hashed_password

    def create_access_token(self, username: str, roles: list) -> str:
        """Create a JWT access token.
        
        Args:
            username: Username
            roles: User roles
            
        Returns:
            JWT token string
        """
        expire = datetime.utcnow() + timedelta(hours=self.token_expire_hours)
        payload = {
            "sub": username,
            "roles": roles,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access_token"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Dict:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access_token":
                raise HTTPException(status_code=401, detail="Invalid token type")
            
            # Check if user is still active
            username = payload.get("sub")
            if username in self.users and not self.users[username]["active"]:
                raise HTTPException(status_code=401, detail="User account is deactivated")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def verify_api_key(self, api_key: str) -> Dict:
        """Verify an API key.
        
        Args:
            api_key: API key string
            
        Returns:
            API key information
            
        Raises:
            HTTPException: If API key is invalid
        """
        if api_key not in self.api_keys:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        key_info = self.api_keys[api_key]
        
        if not key_info["active"]:
            raise HTTPException(status_code=401, detail="API key is deactivated")
        
        return key_info

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User information if authenticated, None otherwise
        """
        if username not in self.users:
            return None
        
        user = self.users[username]
        
        if not user["active"]:
            return None
        
        if not self.verify_password(password, user["password_hash"]):
            return None
        
        return {
            "username": username,
            "roles": user["roles"]
        }

    def require_role(self, required_role: str):
        """Create a dependency that requires a specific role.
        
        Args:
            required_role: Required role name
            
        Returns:
            FastAPI dependency function
        """
        def check_role(user: Dict = Depends(require_auth)):
            if required_role not in user.get("roles", []):
                raise HTTPException(
                    status_code=403, 
                    detail=f"Role '{required_role}' required"
                )
            return user
        
        return check_role

    def create_api_key(self, user: str, roles: list) -> str:
        """Create a new API key.
        
        Args:
            user: Username for the API key
            roles: Roles for the API key
            
        Returns:
            Generated API key
        """
        api_key = f"ohana_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "user": user,
            "roles": roles,
            "active": True,
            "created_at": datetime.now()
        }
        
        self.logger.info(f"Created API key for user: {user}")
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if key was revoked, False if not found
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            self.logger.info(f"Revoked API key: {api_key[:16]}...")
            return True
        return False

    def add_user(self, username: str, password: str, roles: list) -> bool:
        """Add a new user.
        
        Args:
            username: Username
            password: Password
            roles: User roles
            
        Returns:
            True if user was added, False if already exists
        """
        if username in self.users:
            return False
        
        self.users[username] = {
            "password_hash": self._hash_password(password),
            "roles": roles,
            "active": True
        }
        
        self.logger.info(f"Added new user: {username}")
        return True

    def update_user_password(self, username: str, new_password: str) -> bool:
        """Update a user's password.
        
        Args:
            username: Username
            new_password: New password
            
        Returns:
            True if password was updated, False if user not found
        """
        if username not in self.users:
            return False
        
        self.users[username]["password_hash"] = self._hash_password(new_password)
        self.logger.info(f"Updated password for user: {username}")
        return True

    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account.
        
        Args:
            username: Username to deactivate
            
        Returns:
            True if user was deactivated, False if not found
        """
        if username not in self.users:
            return False
        
        self.users[username]["active"] = False
        self.logger.info(f"Deactivated user: {username}")
        return True


# Global auth instance
auth_instance = APIAuth()

# Security scheme
security = HTTPBearer()


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """FastAPI dependency for authentication.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User information
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    
    # Try JWT token first
    try:
        payload = auth_instance.verify_token(token)
        return {
            "username": payload["sub"],
            "roles": payload["roles"],
            "auth_type": "jwt"
        }
    except HTTPException:
        pass
    
    # Try API key
    try:
        key_info = auth_instance.verify_api_key(token)
        return {
            "username": key_info["user"],
            "roles": key_info["roles"],
            "auth_type": "api_key"
        }
    except HTTPException:
        pass
    
    # Authentication failed
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials"
    )


async def require_admin(user: Dict = Depends(require_auth)) -> Dict:
    """FastAPI dependency that requires admin role.
    
    Args:
        user: User information from authentication
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user doesn't have admin role
    """
    if "admin" not in user.get("roles", []):
        raise HTTPException(
            status_code=403,
            detail="Admin role required"
        )
    return user


class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 60):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests: Dict[str, list] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed.
        
        Args:
            identifier: Client identifier (IP, user, etc.)
            
        Returns:
            True if request is allowed
        """
        now = time.time()
        
        # Initialize if new identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


# Global rate limiter
rate_limiter = RateLimiter()


async def rate_limit(request: Request) -> None:
    """FastAPI dependency for rate limiting.
    
    Args:
        request: HTTP request
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )