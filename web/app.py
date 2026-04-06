"""FastAPI web application for TabFlow.ai."""

import asyncio
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
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
# Live Recording & Transcription
# ---------------------------------------------------------------------------


@app.get("/record", response_class=HTMLResponse)
async def record_page(request: Request):
    """Live recording page with interactive tab editor."""
    return _render(request, "record.html")


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """WebSocket endpoint for real-time audio transcription.

    Client sends audio chunks (WAV bytes), server returns note events + tab.
    """

    await websocket.accept()
    logger.info("WebSocket transcription session started")

    accumulated_notes = []
    chunk_count = 0

    try:
        while True:
            # Receive audio chunk as bytes
            audio_bytes = await websocket.receive_bytes()
            chunk_count += 1

            # Save chunk to temp file and transcribe
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                from tab_ripper.tabber import STANDARD_TUNING, assign_frets, filter_notes
                from tab_ripper.transcriber import transcribe

                # Transcribe this chunk
                midi_data, note_events = await asyncio.to_thread(
                    transcribe,
                    tmp_path,
                    backend="basic-pitch",
                )

                # Filter and assign frets
                if note_events:
                    filtered = filter_notes(note_events, tuning=STANDARD_TUNING)
                    events = assign_frets(filtered, tuning=STANDARD_TUNING)

                    # Build response with note data
                    notes_json = []
                    for evt in events:
                        for note in evt.notes:
                            notes_json.append(
                                {
                                    "time": round(evt.time, 3),
                                    "string": note.string,
                                    "fret": note.fret,
                                    "pitch": note.midi_pitch,
                                    "duration": round(note.duration, 3),
                                    "technique": "normal",
                                }
                            )

                    accumulated_notes.extend(notes_json)

                    await websocket.send_json(
                        {
                            "type": "notes",
                            "chunk": chunk_count,
                            "notes": notes_json,
                            "total_notes": len(accumulated_notes),
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "empty",
                            "chunk": chunk_count,
                        }
                    )

            finally:
                import os

                os.unlink(tmp_path)

    except Exception as e:
        logger.info("WebSocket session ended: %s", e)


@app.post("/api/jobs/{job_id}/correct")
async def correct_note(job_id: str, request: Request):
    """Save a user's note correction for training."""
    body = await request.json()
    note_index = body.get("note_index")
    corrected_string = body.get("string")
    corrected_fret = body.get("fret")
    corrected_technique = body.get("technique", "normal")

    # Save correction to database
    from sqlalchemy import text

    from .models import async_session

    async with async_session() as session:
        # Store in a corrections log (we'll create the table on first use)
        await session.execute(
            text("""
                INSERT OR IGNORE INTO corrections (job_id, note_index, corrected_string, corrected_fret, corrected_technique, created_at)
                VALUES (:job_id, :note_index, :string, :fret, :technique, :created_at)
            """),
            {
                "job_id": job_id,
                "note_index": note_index,
                "string": corrected_string,
                "fret": corrected_fret,
                "technique": corrected_technique,
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        await session.commit()

    return {"status": "saved", "note_index": note_index}


@app.post("/api/record/save")
async def save_recording(request: Request):
    """Save a completed recording session with its transcription and corrections."""
    body = await request.json()
    notes = body.get("notes", [])
    corrections = body.get("corrections", {})

    # Create a new job-like entry for the recording
    job_id = str(uuid.uuid4())
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    import json

    # Save the corrected notes as training data
    training_data = {
        "source": "live_recording",
        "job_id": job_id,
        "note_count": len(notes),
        "correction_count": len(corrections),
        "notes": notes,
        "corrections": corrections,
    }

    output_path = job_dir / "corrected_training_data.json"
    output_path.write_text(json.dumps(training_data, indent=2))

    # Also save to the training data directory
    corrections_dir = Path("data/corrections")
    corrections_dir.mkdir(parents=True, exist_ok=True)
    (corrections_dir / f"{job_id}.json").write_text(json.dumps(training_data, indent=2))

    return {
        "status": "saved",
        "job_id": job_id,
        "notes": len(notes),
        "corrections": len(corrections),
    }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


@app.get("/jobs/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    file_path = settings.upload_dir / job_id / filename
    if not file_path.exists():
        return HTMLResponse("File not found", status_code=404)
    return FileResponse(file_path, filename=filename, media_type="application/octet-stream")
