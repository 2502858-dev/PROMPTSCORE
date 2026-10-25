
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from authservice.routes import app as auth_router
from leaderbor.api2 import router as leaderbor_router
from leaderbor.api1 import app as leaderbor_api_router

app = FastAPI(title="PromptScore AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(leaderbor_api_router)
app.include_router(leaderbor_router)


@app.get("/health")
async def home():
    return {
        "status": "running",
        "service": "PromptScore AI"
    }