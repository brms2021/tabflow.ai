"""CLI entry point for tab-ripper.

Usage:
    tab-ripper path/to/song.mp3
    tab-ripper path/to/song.wav --output tabs/
    tab-ripper path/to/song.mp3 --model htdemucs_6s --resolution 0.08
"""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from .separator import check_audio_format, get_guitar_stem, separate
from .tabber import (
    assign_frets,
    filter_notes,
    format_tab_header,
    parse_tuning,
    render_ascii_tab,
    tuning_freq_range,
)
from .transcriber import transcribe

load_dotenv()
logger = logging.getLogger("tab_ripper")


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the entire tab_ripper package."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root = logging.getLogger("tab_ripper")
    root.setLevel(level)
    root.addHandler(handler)


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path), nargs=-1, required=True)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for stems, MIDI, and tab files.",
)
@click.option(
    "--model", "-m", default="htdemucs", help="Demucs model: htdemucs (4-stem) or htdemucs_6s (6-stem with guitar)."
)
@click.option(
    "--onset-threshold",
    default=0.6,
    type=float,
    help="Note onset confidence threshold (0-1). Higher = fewer ghost notes.",
)
@click.option("--frame-threshold", default=0.4, type=float, help="Frame activation threshold (0-1).")
@click.option(
    "--resolution",
    "-r",
    default=0.1,
    type=float,
    help="Time resolution in seconds per tab column. Lower = more detail.",
)
@click.option(
    "--tuning",
    "-t",
    default="standard",
    help="Guitar tuning. Presets: standard, drop-d, d, 7-string, drop-a7. "
    "Or comma-separated notes like 'B1,E2,A2,D3,G3,B3,E4'.",
)
@click.option(
    "--amplitude-threshold",
    default=0.4,
    type=float,
    help="Note amplitude/confidence filter (0-1). Higher = fewer notes, less noise.",
)
@click.option("--min-note-length", default=50.0, type=float, help="Minimum note duration in ms (filters noise blips).")
@click.option("--pdf/--no-pdf", default=True, help="Generate PDF tablature output.")
@click.option("--llm/--no-llm", default=True, help="Use Claude to refine fret assignments and identify techniques.")
@click.option("--llm-model", default="claude-sonnet-4-6", help="Claude model for technique analysis.")
@click.option(
    "--llm-max-phrases",
    default=20,
    type=int,
    help="Max number of phrases to send to LLM (caps API calls). 0 = unlimited.",
)
@click.option(
    "--skip-separation", is_flag=True, help="Skip Demucs and use the input file directly (if already isolated)."
)
@click.option("--width", "-w", default=80, type=int, help="Tab line width in characters.")
@click.option("--bpm/--no-bpm", default=False, help="Detect tempo and add bar lines to output.")
@click.option("--gp5", is_flag=True, default=False, help="Export to Guitar Pro 5 (.gp5) format.")
@click.option("--quantize", default=0.0, type=float, help="Beat quantization strength (0-1). 0=off, 1=snap to beat.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    audio_file: tuple[Path, ...],
    output: Path | None,
    model: str,
    onset_threshold: float,
    frame_threshold: float,
    resolution: float,
    tuning: str,
    amplitude_threshold: float,
    min_note_length: float,
    pdf: bool,
    llm: bool,
    llm_model: str,
    llm_max_phrases: int,
    skip_separation: bool,
    width: int,
    bpm: bool,
    gp5: bool,
    quantize: float,
    verbose: bool,
):
    """Rip guitar tablature from one or more audio files.

    Takes a full mix (MP3, WAV, FLAC, etc.), isolates the guitar,
    transcribes to MIDI, and generates ASCII tablature + PDF.
    """
    _setup_logging(verbose)

    if len(audio_file) > 1:
        click.echo(f"Processing {len(audio_file)} files...")

    errors = []
    for file_idx, single_file in enumerate(audio_file):
        if len(audio_file) > 1:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"  File {file_idx + 1}/{len(audio_file)}: {single_file.name}")
            click.echo(f"{'=' * 60}")
        try:
            _process_file(
                single_file,
                output=output,
                model=model,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                resolution=resolution,
                tuning=tuning,
                amplitude_threshold=amplitude_threshold,
                min_note_length=min_note_length,
                pdf=pdf,
                llm=llm,
                llm_model=llm_model,
                llm_max_phrases=llm_max_phrases,
                skip_separation=skip_separation,
                width=width,
                bpm=bpm,
                gp5=gp5,
                quantize=quantize,
            )
        except Exception as e:
            logger.error("Failed to process %s: %s", single_file.name, e)
            errors.append((single_file.name, str(e)))

    if len(audio_file) > 1:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"  Batch complete: {len(audio_file) - len(errors)}/{len(audio_file)} succeeded")
        if errors:
            click.echo("  Errors:")
            for name, err in errors:
                click.echo(f"    - {name}: {err}")
        click.echo(f"{'=' * 60}")


def _process_file(
    audio_file: Path,
    *,
    output: Path | None,
    model: str,
    onset_threshold: float,
    frame_threshold: float,
    resolution: float,
    tuning: str,
    amplitude_threshold: float,
    min_note_length: float,
    pdf: bool,
    llm: bool,
    llm_model: str,
    llm_max_phrases: int,
    skip_separation: bool,
    width: int,
    bpm: bool,
    gp5: bool,
    quantize: float,
) -> None:
    """Process a single audio file through the full pipeline."""
    # Parse tuning
    tuning_pitches, string_names = parse_tuning(tuning)
    num_strings = len(tuning_pitches)
    min_freq, max_freq = tuning_freq_range(tuning_pitches)
    logger.info("Tuning: %s (%d-string)", " ".join(string_names), num_strings)

    if output is None:
        output = audio_file.parent / "tab-ripper-output"
    output.mkdir(parents=True, exist_ok=True)

    track_name = audio_file.stem

    # Validate audio format
    check_audio_format(audio_file)

    # --- Step 1: Source separation ---
    if skip_separation:
        logger.info("Skipping separation — using %s directly", audio_file.name)
        guitar_path = audio_file
    else:
        logger.info("Step 1/4: Separating stems from %s...", audio_file.name)
        stems = separate(audio_file, output_dir=output / "stems", model=model)
        guitar_path = get_guitar_stem(stems)
        logger.info("Guitar stem: %s", guitar_path)

    # --- Step 2: Transcribe to MIDI ---
    logger.info("Step 2/4: Transcribing guitar to MIDI...")
    midi_path = output / f"{track_name}.mid"
    midi_data, note_events = transcribe(
        guitar_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        midi_output_path=midi_path,
    )

    # --- Step 3: Filter notes ---
    logger.info("Step 3/4: Filtering notes...")
    filtered_notes = filter_notes(
        note_events,
        tuning=tuning_pitches,
        amplitude_threshold=amplitude_threshold,
        min_duration_ms=min_note_length,
    )
    logger.info("%d notes after filtering", len(filtered_notes))

    # --- Step 4a: Assign frets (Viterbi) ---
    logger.info("Step 4/5: Assigning fret positions (Viterbi)...")
    events = assign_frets(filtered_notes, tuning=tuning_pitches)

    # --- Step 4b: LLM technique analysis ---
    annotations = None
    if llm:
        logger.info("Step 4b/5: Refining with LLM technique analysis...")
        from .llm_analyzer import analyze_and_refine

        events, annotations = analyze_and_refine(
            events,
            tuning=tuning_pitches,
            string_names=string_names,
            model=llm_model,
            max_phrases=llm_max_phrases if llm_max_phrases > 0 else None,
        )

    # --- Step 4c: BPM detection and quantization ---
    detected_bpm = None
    bar_lines = None
    if bpm:
        logger.info("Detecting tempo...")
        from .tempo import detect_tempo, generate_bar_lines, quantize_events

        detected_bpm, beat_times = detect_tempo(str(guitar_path))
        bar_lines = generate_bar_lines(beat_times)
        logger.info("%.1f BPM, %d bar lines", detected_bpm, len(bar_lines))

        if quantize > 0:
            logger.info("Quantizing events (strength=%.2f)...", quantize)
            events = quantize_events(events, beat_times, strength=quantize)

    # --- Step 5: Generate tablature ---
    logger.info("Step 5/5: Generating tablature...")

    note_count = sum(len(e.notes) for e in events)
    duration = max(e.time for e in events) if events else 0.0

    header = format_tab_header(
        track_name,
        note_count,
        duration,
        tuning=tuning_pitches,
        string_names=string_names,
    )
    if detected_bpm:
        header = header.rstrip("\n") + f"\n  BPM: {detected_bpm:.0f} | Bars: {len(bar_lines)}\n{'=' * 60}\n"
    tab_text = render_ascii_tab(
        events,
        tuning=tuning_pitches,
        string_names=string_names,
        columns_per_line=width,
        time_resolution=resolution,
    )

    full_tab = header + "\n" + tab_text

    # Save ASCII tab
    tab_path = output / f"{track_name}.tab.txt"
    tab_path.write_text(full_tab)

    # Print tab to console
    click.echo(f"\n{full_tab}")

    # Generate PDF
    pdf_path = None
    if pdf:
        from .pdf_renderer import render_pdf

        pdf_path = output / f"{track_name}.tab.pdf"
        render_pdf(
            events,
            pdf_path,
            title=track_name,
            tuning=tuning_pitches,
            string_names=string_names,
            annotations=annotations,
        )
        logger.info("PDF saved to %s", pdf_path)

    # Generate Guitar Pro 5
    gp5_path = None
    if gp5:
        from .gp_exporter import export_gp5

        gp5_path = output / f"{track_name}.gp5"
        export_gp5(
            events,
            gp5_path,
            title=track_name,
            tuning=tuning_pitches,
            string_names=string_names,
            bpm=detected_bpm or 120.0,
        )

    click.echo("\nDone!")
    click.echo(f"  MIDI:  {midi_path}")
    click.echo(f"  Tab:   {tab_path}")
    if pdf_path:
        click.echo(f"  PDF:   {pdf_path}")
    if gp5_path:
        click.echo(f"  GP5:   {gp5_path}")
    if not skip_separation:
        click.echo(f"  Stems: {output / 'stems'}")


if __name__ == "__main__":
    main()
