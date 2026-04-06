"""FastAPI web application for TabFlow.ai."""

import asyncio
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from .config import settings
from .models import Job, async_session, init_db

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

# Static files and templates
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _render(request: Request, template: str, context: dict | None = None, status_code: int = 200):
    """Render a Jinja2 template compatible with Starlette 1.0."""
    ctx = {"request": request}
    if context:
        ctx.update(context)
    return templates.TemplateResponse(request, template, ctx, status_code=status_code)


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("TabFlow.ai web app started")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    async with async_session() as session:
        result = await session.execute(select(Job).order_by(Job.created_at.desc()).limit(10))
        recent_jobs = result.scalars().all()

    return _render(
        request,
        "index.html",
        {
            "recent_jobs": recent_jobs,
            "tuning_presets": ["standard", "drop-d", "d", "7-string", "drop-a7"],
            "max_upload_mb": settings.max_upload_mb,
        },
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            return _render(request, "error.html", {"message": "Job not found"}, status_code=404)

    return _render(request, "job.html", {"job": job})


@app.get("/jobs/{job_id}/progress", response_class=HTMLResponse)
async def job_progress(request: Request, job_id: str):
    async with async_session() as session:
        job = await session.get(Job, job_id)
        if not job:
            return HTMLResponse("<p>Job not found</p>", status_code=404)

    return _render(request, "partials/progress.html", {"job": job})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    async with async_session() as session:
        result = await session.execute(select(Job).order_by(Job.created_at.desc()).limit(50))
        jobs = result.scalars().all()

    return _render(request, "dashboard.html", {"jobs": jobs})


# ---------------------------------------------------------------------------
# Upload & Processing
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_file(
    audio_file: UploadFile = File(...),
    tuning: str = Form("standard"),
    model: str = Form("htdemucs_ft"),
    enable_llm: bool = Form(False),
):
    job_id = str(uuid.uuid4())
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / audio_file.filename
    with open(input_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    async with async_session() as session:
        job = Job(
            id=job_id,
            filename=audio_file.filename,
            tuning=tuning,
            model=model,
            enable_llm=int(enable_llm),
            status="pending",
        )
        session.add(job)
        await session.commit()

    asyncio.create_task(_run_pipeline(job_id, input_path, tuning, model, bool(enable_llm)))
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


async def _run_pipeline(job_id: str, input_path: Path, tuning: str, model: str, enable_llm: bool):
    """Run the tab-ripper pipeline in the background."""
    job_dir = settings.upload_dir / job_id

    async def update_job(**kwargs):
        async with async_session() as session:
            job = await session.get(Job, job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                await session.commit()

    await update_job(status="processing", step="Starting...", progress=0.0)

    try:
        from tab_ripper.separator import check_audio_format
        from tab_ripper.tabber import (
            assign_frets,
            filter_notes,
            format_tab_header,
            parse_tuning,
            render_ascii_tab,
            tuning_freq_range,
        )
        from tab_ripper.transcriber import transcribe

        check_audio_format(input_path)
        tuning_pitches, string_names = parse_tuning(tuning)
        min_freq, max_freq = tuning_freq_range(tuning_pitches)
        track_name = input_path.stem

        await update_job(step="Transcribing audio to MIDI...", progress=10.0)
        midi_data, note_events = await asyncio.to_thread(
            transcribe,
            str(input_path),
            backend="basic-pitch",
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            midi_output_path=str(job_dir / f"{track_name}.mid"),
        )

        await update_job(step="Filtering notes...", progress=50.0)
        filtered = await asyncio.to_thread(filter_notes, note_events, tuning=tuning_pitches)

        await update_job(step="Assigning fret positions...", progress=65.0)
        events = await asyncio.to_thread(assign_frets, filtered, tuning=tuning_pitches)

        annotations = None
        if enable_llm:
            await update_job(step="Analyzing techniques (LLM)...", progress=75.0)
            from tab_ripper.llm_analyzer import analyze_and_refine

            events, annotations = await asyncio.to_thread(
                analyze_and_refine,
                events,
                tuning=tuning_pitches,
                string_names=string_names,
                max_phrases=10,
            )

        await update_job(step="Generating tablature...", progress=90.0)
        note_count = sum(len(e.notes) for e in events)
        duration = max(e.time for e in events) if events else 0.0

        header = format_tab_header(track_name, note_count, duration, tuning=tuning_pitches, string_names=string_names)
        tab_text = render_ascii_tab(events, tuning=tuning_pitches, string_names=string_names)
        tab_path = job_dir / f"{track_name}.tab.txt"
        tab_path.write_text(header + "\n" + tab_text)

        from tab_ripper.pdf_renderer import render_pdf

        pdf_path = job_dir / f"{track_name}.tab.pdf"
        await asyncio.to_thread(
            render_pdf,
            events,
            pdf_path,
            title=track_name,
            tuning=tuning_pitches,
            string_names=string_names,
            annotations=annotations,
        )

        from tab_ripper.gp_exporter import export_gp5

        gp5_path = job_dir / f"{track_name}.gp5"
        await asyncio.to_thread(
            export_gp5, events, gp5_path, title=track_name, tuning=tuning_pitches, string_names=string_names
        )

        await update_job(
            status="completed",
            step="Done!",
            progress=100.0,
            note_count=note_count,
            output_midi=f"{track_name}.mid",
            output_tab=f"{track_name}.tab.txt",
            output_pdf=f"{track_name}.tab.pdf",
            output_gp5=f"{track_name}.gp5",
            completed_at=datetime.utcnow(),
        )
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        await update_job(status="failed", step="Error", error=str(e))


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    file_path = settings.upload_dir / job_id / filename
    if not file_path.exists():
        return HTMLResponse("File not found", status_code=404)
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")
