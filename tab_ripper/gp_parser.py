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
    """Parse GPX (Guitar Pro 6) files.

    GPX can be either:
    - ZIP archive containing XML (newer exports)
    - BCFZ binary-compressed format (Guitar Pro 6 native)
    """
    # Check if it's a ZIP or BCFZ format
    with open(file_path, "rb") as f:
        header = f.read(4)

    if header[:2] == b"PK":
        # Standard ZIP
        return _parse_gp_xml_archive(file_path, score_path="Content/score.gpif")
    elif header == b"BCFZ":
        # Binary compressed GPX — need to decompress
        return _parse_gpx_bcfz(file_path)
    else:
        raise ValueError(f"Unknown GPX format (header: {header!r}). Expected ZIP or BCFZ.")


def _decompress_bcfz(data: bytes) -> bytes:
    """Decompress BCFZ (Guitar Pro 6) to raw BCFS filesystem bytes.

    Ported from TuxGuitar's GPXFileSystem.java (v6) and the Rust GPX reader.

    Format:
    - Bytes 0-3: magic 'BCFZ'
    - Bytes 4-7: 32-bit LE expected decompressed size
    - Bytes 8+:  single bitstream containing interleaved literal and
                 back-reference chunks (LZ77-style, bit-level encoding).

    Bit encoding (MSB-first within each byte):
    - flag bit = 0: literal (uncompressed) chunk
        - next 2 bits (reversed / LSB-first): byte count
        - then read that many literal bytes (each 8 bits, MSB-first)
    - flag bit = 1: back-reference (compressed) chunk
        - next 4 bits (MSB-first): word_size
        - next word_size bits (reversed / LSB-first): offset
        - next word_size bits (reversed / LSB-first): length
        - copy min(length, offset) bytes from (current_pos - offset) in output
    """
    import struct

    if data[:4] != b"BCFZ":
        raise ValueError("Not a BCFZ file")

    expected_size = struct.unpack_from("<I", data, 4)[0]
    src = _GPXBitReader(data[8:])
    result = bytearray()

    while not src.end() and len(result) < expected_size:
        flag = src.read_bits(1)
        if flag == 1:
            # Compressed back-reference
            word_size = src.read_bits(4)
            offset = src.read_bits_reversed(word_size)
            size = src.read_bits_reversed(word_size)

            buf_snapshot = bytes(result)
            copy_src = len(buf_snapshot) - offset
            count = min(size, offset) if offset > 0 else 0
            for i in range(count):
                result.append(buf_snapshot[copy_src + i])
        else:
            # Uncompressed literal bytes
            size = src.read_bits_reversed(2)
            for _ in range(size):
                byte_val = src.read_bits(8)
                if byte_val < 0:
                    break
                result.append(byte_val & 0xFF)

    return bytes(result[:expected_size])


class _GPXBitReader:
    """Bit-level reader matching TuxGuitar's GPXByteBuffer.

    Reads bits MSB-first from each byte (bit 7 first, bit 0 last).
    """

    def __init__(self, buf: bytes):
        self._buf = buf
        self._pos = 0  # position in bits

    def end(self) -> bool:
        return (self._pos // 8) >= len(self._buf)

    def read_bit(self) -> int:
        byte_idx = self._pos // 8
        bit_offset = 7 - (self._pos % 8)  # MSB-first
        if byte_idx < 0 or byte_idx >= len(self._buf):
            return -1
        bit = (self._buf[byte_idx] >> bit_offset) & 0x01
        self._pos += 1
        return bit

    def read_bits(self, count: int) -> int:
        """Read *count* bits MSB-first (big-endian)."""
        bits = 0
        for i in range(count - 1, -1, -1):
            bits |= self.read_bit() << i
        return bits

    def read_bits_reversed(self, count: int) -> int:
        """Read *count* bits LSB-first (little-endian)."""
        bits = 0
        for i in range(count):
            bits |= self.read_bit() << i
        return bits

    def read_bytes(self, count: int) -> bytes:
        return bytes(self.read_bits(8) for _ in range(count))


def _parse_gpx_bcfz(file_path: Path) -> GPTrack:
    """Parse a BCFZ-compressed GPX file.

    Decompresses to get the BCFS internal filesystem, then extracts score.gpif XML.
    """
    logger.info("Parsing BCFZ-compressed GPX: %s", file_path.name)

    with open(file_path, "rb") as f:
        raw = f.read()

    decompressed = _decompress_bcfz(raw)

    # Extract score XML from the BCFS filesystem.
    # Strategy 1: Try structured filesystem extraction
    score_xml = None
    files = _extract_bcfs_filesystem(decompressed)
    for name, content in files.items():
        if "score.gpif" in name.lower() or name.lower().endswith(".gpif"):
            score_xml = content
            break

    # Strategy 2: Scan for GPIF XML directly in the decompressed data
    if score_xml is None:
        gpif_start = decompressed.find(b"<GPIF")
        if gpif_start >= 0:
            gpif_end = decompressed.find(b"</GPIF>", gpif_start)
            if gpif_end >= 0:
                score_xml = decompressed[gpif_start : gpif_end + 7]
                logger.debug("Found GPIF XML at offset %d (%d bytes)", gpif_start, len(score_xml))

    if score_xml is None:
        logger.error("Could not find GPIF XML in BCFS data")

    result = GPTrack(title=file_path.stem)

    if score_xml is None:
        return result

    try:
        root = ET.fromstring(score_xml)
        return _parse_gpif_xml(root, result)
    except ET.ParseError as e:
        logger.error("Failed to parse XML from BCFZ: %s", e)
        return result


def _extract_bcfs_filesystem(data: bytes) -> dict[str, bytes]:
    """Extract files from a decompressed BCFS filesystem image.

    Ported from TuxGuitar's GPXFileSystem.java (the HEADER_BCFS branch).

    Layout:
    - The decompressed output starts with 'BCFS' magic (4 bytes) consumed
      during the recursive call from _decompress_bcfz, but the first sector
      (0x1000 bytes) is the superblock / header.
    - Directory entries start at sector 1 (offset 0x1000) and repeat every
      0x1000 bytes.
    - A directory entry with int32 == 2 at its start is a file entry:
        offset +0x04: filename (127 bytes, null-terminated)
        offset +0x8C: file size (int32 LE)
        offset +0x94: array of int32 LE sector indices (block map),
                      terminated by 0
    - File data lives in the sectors pointed to by the block map.
    """
    import struct

    SECTOR_SIZE = 0x1000

    def _get_int(buf: bytes, off: int) -> int:
        if off + 4 > len(buf):
            return 0
        return struct.unpack_from("<I", buf, off)[0]

    def _get_string(buf: bytes, off: int, max_len: int) -> str:
        chars = []
        for i in range(max_len):
            if off + i >= len(buf):
                break
            b = buf[off + i]
            if b == 0:
                break
            chars.append(chr(b))
        return "".join(chars)

    def _get_bytes(buf: bytes, off: int, length: int) -> bytes:
        end = min(off + length, len(buf))
        result = buf[off:end]
        # Pad with zeros if source is shorter
        if len(result) < length:
            result += b"\x00" * (length - len(result))
        return result

    files: dict[str, bytes] = {}

    # Walk directory entries starting at the second sector
    offset = SECTOR_SIZE
    while offset + 3 < len(data):
        entry_type = _get_int(data, offset)
        if entry_type == 2:
            # File entry
            idx_filename = offset + 4
            idx_filesize = offset + 0x8C
            idx_blocks = offset + 0x94

            filename = _get_string(data, idx_filename, 127)
            file_size = _get_int(data, idx_filesize)

            # Read sector chain and collect file data
            file_data = bytearray()
            block_count = 0
            while True:
                block = _get_int(data, idx_blocks + 4 * block_count)
                if block == 0:
                    break
                sector_off = block * SECTOR_SIZE
                file_data.extend(_get_bytes(data, sector_off, SECTOR_SIZE))
                block_count += 1

            # Trim to actual file size
            if len(file_data) >= file_size:
                files[filename] = bytes(file_data[:file_size])
            else:
                logger.warning(
                    "BCFS: file '%s' expected %d bytes but got %d",
                    filename,
                    file_size,
                    len(file_data),
                )
                files[filename] = bytes(file_data)

        offset += SECTOR_SIZE

    return files


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

    root = ET.fromstring(xml_data)
    return _parse_gpif_xml(root, result)


def _parse_gpif_xml(root: ET.Element, result: GPTrack) -> GPTrack:
    """Parse a GPIF XML element tree into a GPTrack."""

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
