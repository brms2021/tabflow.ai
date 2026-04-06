"""SQLAlchemy models for TabFlow.ai web app."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    step = Column(String(100), default="")
    filename = Column(String(255), nullable=False)
    tuning = Column(String(50), default="standard")
    model = Column(String(50), default="htdemucs_ft")
    transcriber = Column(String(50), default="basic-pitch")
    enable_llm = Column(Integer, default=1)
    note_count = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Output file paths (relative to upload_dir/job_id/)
    output_midi = Column(String(255), nullable=True)
    output_tab = Column(String(255), nullable=True)
    output_pdf = Column(String(255), nullable=True)
    output_gp5 = Column(String(255), nullable=True)


class Correction(Base):
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), nullable=False)
    note_index = Column(Integer, nullable=False)
    corrected_string = Column(Integer, nullable=False)
    corrected_fret = Column(Integer, nullable=False)
    corrected_technique = Column(String(50), default="normal")
    created_at = Column(DateTime, default=datetime.utcnow)


engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    async with async_session() as session:
        yield session
