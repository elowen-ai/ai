import os
import jwt
import asyncio
from functools import wraps
from dotenv import load_dotenv
from typing import Any, Callable

load_dotenv()

JWT_SECRET: str | None = os.getenv("JWT_SECRET")

class Auth:
    sessions: list = []

    def authenticate(self, sid, authData) -> dict[str, Any]:
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

    def isAuthenticated(self, f: Callable) -> Callable:
        @wraps(f)
        async def async_wrapper(sid, *args, **kwargs) -> Any:
            if sid not in self.sessions:
                return {"status": False, "message": "Not authenticated"}
            return await f(sid, *args, **kwargs)

        @wraps(f)
        def sync_wrapper(sid, *args, **kwargs) -> Any:
            if sid not in self.sessions:
                return {"status": False, "message": "Not authenticated"}
            return f(sid, *args, **kwargs)

        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        return sync_wrapper