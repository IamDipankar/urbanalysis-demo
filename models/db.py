import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Prefer SQLite for local testing unless an explicit DATABASE_URL is provided
DATABASE_URL = os.getenv("DATABASE_URL")
USE_SQLITE = os.getenv("USE_SQLITE", "1").strip() not in ("0", "false", "False")

if not DATABASE_URL and USE_SQLITE:
    DATABASE_URL = "sqlite:///./local_fallback.db"

# Create engine with sensible defaults
engine = create_engine(
    DATABASE_URL or "sqlite:///./local_fallback.db",
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# Dependency helper for FastAPI routes (sync context)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
