from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.analyze import router as analyze_router


app = FastAPI(title="Skillforge Resume Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "message": "Skillforge Resume Analyzer API is running.",
        "health_url": "/health",
        "analyze_url": "/analyze"
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
