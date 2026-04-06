"""Parse Guitar Pro files (.gp3/.gp4/.gp5/.gpx/.gp) for ground truth extraction.

Supports:
- .gp3, .gp4, .gp5: via pyguitarpro (binary format)
- .gpx: Guitar Pro 6 (ZIP + XML)
- .gp: Guitar Pro 7/8 (ZIP + XML)

Extracts per-note: string, fret, timing, technique, with exact beat positions.
This is the highest-quality ground truth source — no OCR or spatial guessing.
"""

import json
import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GPNote:
    """A single note from a Guitar Pro file."""

    bar: int
    beat: float  # beat position within bar (0-based)
    string: int  # 0=lowest
    fret: int
    pitch_midi: int
    duration_beats: float
    technique: str = "normal"
    velocity: int = 100
    bend: float = 0.0  # bend value in semitones
    slide: bool = False
    hammer_on: bool = False
    pull_off: bool = False
    vibrato: bool = False
    palm_mute: bool = False
    harmonic: bool = False
    tap: bool = False


@dataclass
class GPTrack:
    """A parsed Guitar Pro track."""

    title: str = ""
    artist: str = ""
    bpm: float = 120.0
    tuning: list[int] = field(default_factory=lambda: [40, 45, 50, 55, 59, 64])
    tuning_name: str = "standard"
    num_strings: int = 6
    time_sig_num: int = 4
    time_sig_den: int = 4
    notes: list[GPNote] = field(default_factory=list)
    num_bars: int = 0


def parse_gp_file(file_path: str | Path) -> GPTrack:
    """Parse any Guitar Pro format file.

    Auto-detects format by extension and delegates to appropriate parser.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext in (".gp3", ".gp4", ".gp5"):
        return _parse_gp345(file_path)
    elif ext == ".gpx":
        return _parse_gpx(file_path)
    elif ext == ".gp":
        return _parse_gp7(file_path)
    else:
        raise ValueError(f"Unsupported Guitar Pro format: {ext}. Use .gp3/.gp4/.gp5/.gpx/.gp")


def _parse_gp345(file_path: Path) -> GPTrack:
    """Parse GP3/4/5 files using pyguitarpro."""
    import guitarpro

    logger.info("Parsing GP3/4/5: %s", file_path.name)
    song = guitarpro.parse(str(file_path))

    result = GPTrack(
        title=song.title or file_path.stem,
        artist=song.artist or "",
        bpm=float(song.tempo.value) if song.tempo else 120.0,
    )

    # Find the guitar track (first non-percussion track)
    guitar_track = None
    for track in song.tracks:
        if not track.isPercussionTrack:
            guitar_track = track
            break

    if guitar_track is None:
        logger.warning("No guitar track found in %s", file_path.name)
        return result

    # Extract tuning
    result.num_strings = len(guitar_track.strings)
    result.tuning = [s.value for s in reversed(guitar_track.strings)]  # GP is high-to-low, we want low-to-high

    result.num_bars = len(guitar_track.measures)

    # Extract notes
    for bar_idx, measure in enumerate(guitar_track.measures):
        for voice in measure.voices:
            beat_pos = 0.0
            for beat in voice.beats:
                dur_beats = 4.0 / beat.duration.value
                if beat.duration.isDotted:
                    dur_beats *= 1.5
                if beat.duration.tuplet:
                    dur_beats *= beat.duration.tuplet.enters / beat.duration.tuplet.times

                for note in beat.notes:
                    if note.type == guitarpro.NoteType.rest:
                        continue

                    # String: GP is 1-based high-to-low, convert to 0-based low-to-high
                    string_idx = result.num_strings - note.string

                    gp_note = GPNote(
                        bar=bar_idx + 1,
                        beat=round(beat_pos, 4),
                        string=string_idx,
                        fret=note.value,
                        pitch_midi=result.tuning[string_idx] + note.value if string_idx < len(result.tuning) else 0,
                        duration_beats=round(dur_beats, 4),
                        velocity=note.velocity,
                    )

                    # Extract techniques from note effects
                    if note.effect:
                        eff = note.effect
                        if eff.bend:
                            gp_note.bend = eff.bend.value / 100.0 if eff.bend.value else 0.0
                            gp_note.technique = "bend"
                        if eff.slide:
                            gp_note.slide = True
                            gp_note.technique = "slide"
                        if eff.hammer:
                            gp_note.hammer_on = True
                            gp_note.technique = "hammer-on"
                        if eff.harmonic:
                            gp_note.harmonic = True
                            gp_note.technique = "harmonic"
                        if eff.palmMute:
                            gp_note.palm_mute = True
                            gp_note.technique = "palm-mute"
                        if eff.vibrato:
                            gp_note.vibrato = True

                    # Beat-level effects
                    if beat.effect:
                        if beat.effect.tapSlapPop:
                            gp_note.tap = True
                            gp_note.technique = "tap"

                    result.notes.append(gp_note)

                beat_pos += dur_beats

    logger.info("Parsed %d notes across %d bars", len(result.notes), result.num_bars)
    return result


def _parse_gpx(file_path: Path) -> GPTrack:
    """Parse GPX (Guitar Pro 6) files — ZIP containing XML."""
    return _parse_gp_xml_archive(file_path, score_path="Content/score.gpif")


def _parse_gp7(file_path: Path) -> GPTrack:
    """Parse GP (Guitar Pro 7/8) files — ZIP containing XML."""
    return _parse_gp_xml_archive(file_path, score_path="Content/score.gpif")


def _parse_gp_xml_archive(file_path: Path, score_path: str) -> GPTrack:
    """Parse a Guitar Pro ZIP archive (GPX or GP7) containing XML score data."""
    logger.info("Parsing GP XML archive: %s", file_path.name)

    result = GPTrack(title=file_path.stem)

    with zipfile.ZipFile(str(file_path), "r") as zf:
        names = zf.namelist()
        logger.debug("Archive contents: %s", names)

        # Find the score XML file
        score_file = None
        for name in names:
            if name.lower().endswith("score.gpif") or name.lower().endswith(".gpif"):
                score_file = name
                break

        if score_file is None:
            logger.warning("No score.gpif found in %s. Contents: %s", file_path.name, names)
            return result

        xml_data = zf.read(score_file)

    # Parse XML
    root = ET.fromstring(xml_data)

    # Extract metadata
    score_el = root.find(".//Score")
    if score_el is not None:
        title_el = score_el.find("Title")
        artist_el = score_el.find("Artist")
        if title_el is not None and title_el.text:
            result.title = title_el.text
        if artist_el is not None and artist_el.text:
            result.artist = artist_el.text

    # Extract tempo
    master_track = root.find(".//MasterTrack")
    if master_track is not None:
        automations = master_track.findall(".//Automation")
        for auto in automations:
            type_el = auto.find("Type")
            if type_el is not None and type_el.text == "Tempo":
                value_el = auto.find("Value")
                if value_el is not None:
                    try:
                        result.bpm = float(value_el.text.split()[0])
                    except (ValueError, IndexError):
                        pass

    # Extract tracks and tuning
    tracks = root.findall(".//Track")
    for track in tracks:
        # Find first non-drum track
        properties = track.findall(".//Properties/Property")
        is_drum = False
        for prop in properties:
            if prop.get("name") == "PercussionTrack":
                is_drum = True
        if not is_drum:
            # Get tuning
            tuning_el = track.find(".//Properties/Property[@name='Tuning']/Pitches")
            if tuning_el is not None and tuning_el.text:
                pitches = [int(p) for p in tuning_el.text.split()]
                result.tuning = list(reversed(pitches))  # GP XML is high-to-low
                result.num_strings = len(result.tuning)
            break

    # Extract notes from bars/beats
    # GPX XML structure: MasterBars -> MasterBar -> Bars -> Bar -> Voices -> Voice -> Beats -> Beat -> Notes -> Note
    beats_db = {}
    for beat_el in root.findall(".//Beats/Beat"):
        beat_id = beat_el.get("id")
        if beat_id:
            beats_db[beat_id] = beat_el

    notes_db = {}
    for note_el in root.findall(".//Notes/Note"):
        note_id = note_el.get("id")
        if note_id:
            notes_db[note_id] = note_el

    # Walk through bars
    master_bars = root.findall(".//MasterBars/MasterBar")
    result.num_bars = len(master_bars)

    for bar_idx, mbar in enumerate(master_bars):
        bar_ids_el = mbar.find("Bars")
        if bar_ids_el is None or not bar_ids_el.text:
            continue

        bar_ids = bar_ids_el.text.split()
        for bar_id in bar_ids:
            bar_el = root.find(f".//Bars/Bar[@id='{bar_id}']")
            if bar_el is None:
                continue

            voices_el = bar_el.find("Voices")
            if voices_el is None or not voices_el.text:
                continue

            voice_ids = voices_el.text.split()
            for voice_id in voice_ids:
                voice_el = root.find(f".//Voices/Voice[@id='{voice_id}']")
                if voice_el is None:
                    continue

                beat_ids_el = voice_el.find("Beats")
                if beat_ids_el is None or not beat_ids_el.text:
                    continue

                beat_pos = 0.0
                for beat_id in beat_ids_el.text.split():
                    beat_el = beats_db.get(beat_id)
                    if beat_el is None:
                        continue

                    # Get duration
                    dur_el = beat_el.find("Duration")
                    dur_beats = 1.0  # default quarter note
                    if dur_el is not None and dur_el.text:
                        dur_map = {
                            "Whole": 4.0,
                            "Half": 2.0,
                            "Quarter": 1.0,
                            "Eighth": 0.5,
                            "16th": 0.25,
                            "32nd": 0.125,
                        }
                        dur_beats = dur_map.get(dur_el.text, 1.0)

                    # Get notes
                    note_ids_el = beat_el.find("Notes")
                    if note_ids_el is not None and note_ids_el.text:
                        for note_id in note_ids_el.text.split():
                            note_el = notes_db.get(note_id)
                            if note_el is None:
                                continue

                            # Extract string and fret
                            props = note_el.findall("Properties/Property")
                            string_val = -1
                            fret_val = -1
                            for prop in props:
                                name = prop.get("name")
                                fret_el = prop.find("Fret")
                                string_el = prop.find("String")
                                if name == "String" and string_el is not None:
                                    string_val = int(string_el.text)
                                elif name == "Fret" and fret_el is not None:
                                    fret_val = int(fret_el.text)

                            if string_val >= 0 and fret_val >= 0:
                                # GP XML string: 0=highest, convert to 0=lowest
                                string_idx = result.num_strings - 1 - string_val
                                pitch = result.tuning[string_idx] + fret_val if string_idx < len(result.tuning) else 0

                                gp_note = GPNote(
                                    bar=bar_idx + 1,
                                    beat=round(beat_pos, 4),
                                    string=string_idx,
                                    fret=fret_val,
                                    pitch_midi=pitch,
                                    duration_beats=round(dur_beats, 4),
                                )

                                # Check for techniques
                                for prop in props:
                                    name = prop.get("name")
                                    if name == "HammerOn":
                                        gp_note.hammer_on = True
                                        gp_note.technique = "hammer-on"
                                    elif name == "PullOff":
                                        gp_note.pull_off = True
                                        gp_note.technique = "pull-off"
                                    elif name == "Slide":
                                        gp_note.slide = True
                                        gp_note.technique = "slide"
                                    elif name == "Bend":
                                        gp_note.technique = "bend"
                                    elif name == "PalmMute":
                                        gp_note.palm_mute = True
                                        gp_note.technique = "palm-mute"
                                    elif name == "Tap":
                                        gp_note.tap = True
                                        gp_note.technique = "tap"
                                    elif name == "Harmonic":
                                        gp_note.harmonic = True
                                        gp_note.technique = "harmonic"
                                    elif name == "Vibrato":
                                        gp_note.vibrato = True

                                result.notes.append(gp_note)

                    beat_pos += dur_beats

    logger.info("Parsed %d notes across %d bars", len(result.notes), result.num_bars)
    return result


def gp_to_ground_truth_json(track: GPTrack) -> dict:
    """Convert a parsed GP track to ground truth JSON format."""
    return {
        "title": track.title,
        "artist": track.artist,
        "bpm": track.bpm,
        "tuning": track.tuning,
        "tuning_name": track.tuning_name,
        "num_strings": track.num_strings,
        "time_signature": f"{track.time_sig_num}/{track.time_sig_den}",
        "num_bars": track.num_bars,
        "note_count": len(track.notes),
        "notes": [asdict(n) for n in track.notes],
    }


def save_gp_ground_truth(track: GPTrack, output_path: str | Path) -> Path:
    """Save parsed GP data as ground truth JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = gp_to_ground_truth_json(track)
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("GP ground truth saved to %s (%d notes)", output_path, len(track.notes))
    return output_path
