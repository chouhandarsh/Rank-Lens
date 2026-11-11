from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker,declarative_base
from pydantic_settings import BaseSettings,SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    DATABASE_URL: str | None = None
    model_config= SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore"
    )
settings=Settings()


engine = None
if settings.DATABASE_URL:
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
    )
SessionLocal = (
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
    if engine is not None
    else None
)
Base= declarative_base()
def get_db():
    if SessionLocal is None:
        raise RuntimeError("The chatbot requires DATABASE_URL to be set.")
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
def check_db_connection():
    if engine is None:
        return False
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        print("Database connection check failed.")
        return False
