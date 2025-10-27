# import os
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# import dotenv

# # Load environment variables
# dotenv.load_dotenv()

# # Prefer SQLite for local testing unless an explicit DATABASE_URL is provided
# DATABASE_URL = os.getenv("DATABASE_URL")
# USE_SQLITE = os.getenv("USE_SQLITE", "1").strip() not in ("0", "false", "False")

# if not DATABASE_URL or USE_SQLITE:
#     DATABASE_URL = "sqlite:///./local_fallback.db"

# # Create engine with sensible defaults
# engine = create_engine(
#     DATABASE_URL or "sqlite:///./local_fallback.db",
#     pool_pre_ping=True,
#     future=True,
# )

# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
# Base = declarative_base()

# # Dependency helper for FastAPI routes (sync context)
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()



import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import dotenv

# Load environment variables from .env
dotenv.load_dotenv()

# Build PostgreSQL database URL
DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}"
    f"@{os.getenv('DATABASE_HOST')}/{os.getenv('DATABASE_NAME')}"
)

# Create engine with sensible defaults
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

# ORM setup
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# Dependency helper for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
