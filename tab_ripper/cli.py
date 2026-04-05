"""CLI entry point for tab-ripper.

Usage:
    tab-ripper path/to/song.mp3
    tab-ripper path/to/song.wav --output tabs/
    tab-ripper path/to/song.mp3 --model htdemucs_6s --resolution 0.08
"""

from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from .separator import separate, get_guitar_stem
from .transcriber import transcribe
from .tabber import (
    assign_frets, render_ascii_tab, format_tab_header,
    parse_tuning, tuning_freq_range, filter_notes, TUNING_PRESETS,
)


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory for stems, MIDI, and tab files.")
@click.option("--model", "-m", default="htdemucs",
              help="Demucs model: htdemucs (4-stem) or htdemucs_6s (6-stem with guitar).")
@click.option("--onset-threshold", default=0.6, type=float,
              help="Note onset confidence threshold (0-1). Higher = fewer ghost notes.")
@click.option("--frame-threshold", default=0.4, type=float,
              help="Frame activation threshold (0-1).")
@click.option("--resolution", "-r", default=0.1, type=float,
              help="Time resolution in seconds per tab column. Lower = more detail.")
@click.option("--tuning", "-t", default="standard",
              help="Guitar tuning. Presets: standard, drop-d, d, 7-string, drop-a7. "
                   "Or comma-separated notes like 'B1,E2,A2,D3,G3,B3,E4'.")
@click.option("--amplitude-threshold", default=0.4, type=float,
              help="Note amplitude/confidence filter (0-1). Higher = fewer notes, less noise.")
@click.option("--min-note-length", default=50.0, type=float,
              help="Minimum note duration in ms (filters noise blips).")
@click.option("--pdf/--no-pdf", default=True,
              help="Generate PDF tablature output.")
@click.option("--llm/--no-llm", default=True,
              help="Use Claude to refine fret assignments and identify techniques.")
@click.option("--llm-model", default="claude-sonnet-4-6",
              help="Claude model for technique analysis.")
@click.option("--llm-max-phrases", default=20, type=int,
              help="Max number of phrases to send to LLM (caps API calls). 0 = unlimited.")
@click.option("--skip-separation", is_flag=True,
              help="Skip Demucs and use the input file directly (if already isolated).")
@click.option("--width", "-w", default=80, type=int,
              help="Tab line width in characters.")
def main(
    audio_file: Path,
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
):
    """Rip guitar tablature from an audio file.

    Takes a full mix (MP3, WAV, FLAC, etc.), isolates the guitar,
    transcribes to MIDI, and generates ASCII tablature + PDF.
    """
    # Parse tuning
    tuning_pitches, string_names = parse_tuning(tuning)
    num_strings = len(tuning_pitches)
    min_freq, max_freq = tuning_freq_range(tuning_pitches)
    print(f"[tab-ripper] Tuning: {' '.join(string_names)} ({num_strings}-string)")

    if output is None:
        output = audio_file.parent / "tab-ripper-output"
    output.mkdir(parents=True, exist_ok=True)

    track_name = audio_file.stem

    # --- Step 1: Source separation ---
    if skip_separation:
        print(f"\n[tab-ripper] Skipping separation — using {audio_file.name} directly")
        guitar_path = audio_file
    else:
        print(f"\n[tab-ripper] Step 1/4: Separating stems from {audio_file.name}...")
        stems = separate(audio_file, output_dir=output / "stems", model=model)
        guitar_path = get_guitar_stem(stems)
        print(f"[tab-ripper] Guitar stem: {guitar_path}")

    # --- Step 2: Transcribe to MIDI ---
    print(f"\n[tab-ripper] Step 2/4: Transcribing guitar to MIDI...")
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
    print(f"\n[tab-ripper] Step 3/4: Filtering notes...")
    filtered_notes = filter_notes(
        note_events,
        tuning=tuning_pitches,
        amplitude_threshold=amplitude_threshold,
        min_duration_ms=min_note_length,
    )
    print(f"[tab-ripper] {len(filtered_notes)} notes after filtering")

    # --- Step 4a: Assign frets (Viterbi) ---
    print(f"\n[tab-ripper] Step 4/5: Assigning fret positions (Viterbi)...")
    events = assign_frets(filtered_notes, tuning=tuning_pitches)

    # --- Step 4b: LLM technique analysis ---
    annotations = None
    if llm:
        print(f"\n[tab-ripper] Step 4b/5: Refining with LLM technique analysis...")
        from .llm_analyzer import analyze_and_refine
        events, annotations = analyze_and_refine(
            events,
            tuning=tuning_pitches,
            string_names=string_names,
            model=llm_model,
            max_phrases=llm_max_phrases if llm_max_phrases > 0 else None,
        )

    # --- Step 5: Generate tablature ---
    print(f"\n[tab-ripper] Step {'5' if llm else '4'}/{'5' if llm else '4'}: Generating tablature...")

    note_count = sum(len(e.notes) for e in events)
    duration = max(e.time for e in events) if events else 0.0
    step_total = 5 if llm else 4

    header = format_tab_header(
        track_name, note_count, duration,
        tuning=tuning_pitches, string_names=string_names,
    )
    tab_text = render_ascii_tab(
        events, tuning=tuning_pitches, string_names=string_names,
        columns_per_line=width, time_resolution=resolution,
    )

    full_tab = header + "\n" + tab_text

    # Save ASCII tab
    tab_path = output / f"{track_name}.tab.txt"
    tab_path.write_text(full_tab)

    # Print to console
    print(f"\n{full_tab}")

    # Generate PDF
    pdf_path = None
    if pdf:
        from .pdf_renderer import render_pdf
        pdf_path = output / f"{track_name}.tab.pdf"
        render_pdf(
            events, pdf_path,
            title=track_name,
            tuning=tuning_pitches,
            string_names=string_names,
            annotations=annotations,
        )
        print(f"[tab-ripper] PDF saved to {pdf_path}")

    print(f"\n[tab-ripper] Done!")
    print(f"  MIDI:  {midi_path}")
    print(f"  Tab:   {tab_path}")
    if pdf_path:
        print(f"  PDF:   {pdf_path}")
    if not skip_separation:
        print(f"  Stems: {output / 'stems'}")


if __name__ == "__main__":
    main()
