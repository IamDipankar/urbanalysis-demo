from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON

from .db import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_requests"

    id = Column(Integer, primary_key=True, index=True)

    # Business key
    request_id = Column(String(128), unique=True, index=True, nullable=False)

    # Timestamps
    request_coming_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Requested analysis flags
    uhi_hotspot = Column(Boolean, default=False, nullable=False)
    aq_hotspot = Column(Boolean, default=False, nullable=False)
    green_access = Column(Boolean, default=False, nullable=False)

    # Completion durations (seconds since request_coming_time)
    uhi_hotspot_complete_time = Column(Float, nullable=True)
    aq_hotspot_complete_time = Column(Float, nullable=True)
    green_access_complete_time = Column(Float, nullable=True)

    # Results payloads (use generic JSON for cross-database compatibility)
    uhi_result = Column(JSON, nullable=True)
    aq_resutl = Column(JSON, nullable=True)  # Intentionally matching requested name
    green_access_result = Column(JSON, nullable=True)

    # Misc
    any_other_result = Column(JSON, nullable=True)
    remarks = Column(Text, nullable=True)
