"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import websocket
from src.api.routes import auth, courses, users
from src.config.settings import settings

app = FastAPI(
    title="EngageIQ API",
    description="Backend for the EngageIQ intelligent classroom application",
    version="0.1.0",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(courses.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


# TODO: Include route modules
# from src.api.routes import users, courses, sessions, engagement, reports
app.include_router(websocket.router)
