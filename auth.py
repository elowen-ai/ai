import os
import jwt
from typing import Any
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET: str | None = os.getenv("JWT_SECRET")

class Auth:
    sessions: list = []

    def authenticate(self, sid, authData) -> bool:
        if not authData or not sid:
            return { "status": False, "message": "No auth data provided" }
        
        token: str | None = authData.get('token') if authData else None
        if not token: return { "status": False, "message": "No token provided" }

        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return { "status": False, "message": "Token has expired" }
        except jwt.InvalidTokenError:
            return { "status": False, "message": "Invalid token" }
        except Exception as e:
            return { "status": False, "message": f"Authentication error: {str(e)}" }

        self.sessions.append(sid)
        return { "status": True }

    def isAuthenticated(self, f) -> bool:
        @wraps(f)
        def wrapper(sid) -> dict[str, Any] | Any:
            if not sid in self.sessions:
                return { "status": False, "message": "Not authenticated" }
            return f(sid)
        return wrapper
