import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nirf_chatbot.database import check_db_connection
from nirf_chatbot.routes import router

app = FastAPI(title="NIRF Intelligent Chat-Bot", description="Ai powered assistant to answer your queries")
allowed_origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
allowed_origins.extend(
    origin.strip()
    for origin in os.getenv("CHATBOT_CORS_ORIGINS", "").split(",")
    if origin.strip()
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(router)
@app.get("/")
def check_health():
    return {
        "message":"NIRF chatbot is Live"
    }
@app.get("/health")
def health_check():
    return{
        "status":"Heathy",
        "service":"NIRF CHATBOT"
    }
@app.get("/db_health")
def get_db_health():
    is_connected = check_db_connection()
    if is_connected:
        return {
            "status":"connected",
            "databse":"supabse PostgresSQL"
        }
    else :
        return {
            "status":"Not-Connected",
            "database":"Supabase PostgresSQL"
        }


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
    )