#!/usr/bin/env python3

import sys
import os
import re
import json
import shutil
import subprocess
import hashlib
from pathlib import Path

# ============================================================
# LeoYT 2.1 — YouTube Audio → LeoTranscribe → Organized TXT
# ============================================================

HOME = Path.home()
DESKTOP = HOME / "Desktop"
OUTPUT = Path(os.environ.get("LEOTRANSCRIBE_OUTPUT_DIR", str(DESKTOP / "Transcripts"))).expanduser() / "YouTube"
CACHE = Path(os.environ.get("LEOYT_CACHE_DIR", str(HOME / "Library" / "Caches" / "LeoYT"))).expanduser()

def ensure_directory(path):
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Path is not a directory: {path}")

    path.mkdir(parents=True, exist_ok=True)

    if not os.access(path, os.W_OK):
        raise PermissionError(f"No write permission: {path}")

    return path


def safe_name(name):
    name = re.sub(r'[/:\x00-\x1f]', "_", str(name))
    name = name.strip().rstrip(".")
    return name[:150] or "Untitled"


def run(command, capture=False):
    if capture:
        return subprocess.check_output(
            command,
            text=True
        ).strip()

    subprocess.run(command, check=True)


def get_metadata(url):
    raw = run(
        ["yt-dlp", "--flat-playlist", "-J", url],
        capture=True
    )
    return json.loads(raw)


def get_audio_source(work):
    files = [
        p for p in work.glob("source.*")
        if p.is_file() and p.stat().st_size > 0
    ]
    return files[0] if files else None


def transcribe(video_id, title, destination):
    ensure_directory(destination.parent)

    if destination.exists():
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"✓ SKIP (already completed): {destination.name}")
            return True

        print(f"✗ Destination exists but is invalid: {destination}")
        return False

    work = ensure_directory(CACHE / video_id)
    url = f"https://www.youtube.com/watch?v={video_id}"

    audio = work / f"{safe_name(title)}.wav"
    transcript = audio.with_suffix(".txt")

    try:
        # Step 1: Reuse completed transcript if available
        if not transcript.is_file() or transcript.stat().st_size == 0:

            # Step 2: Download audio only if needed
            source = get_audio_source(work)

            if source is None:
                print("▶ Downloading audio...")
                run([
                    "yt-dlp",
                    "--no-playlist",
                    "-f", "bestaudio/best",
                    "-o", str(work / "source.%(ext)s"),
                    url
                ])

                source = get_audio_source(work)

            if source is None:
                raise RuntimeError("Downloaded audio not found")

            # Step 3: Convert to WAV
            if not audio.is_file() or audio.stat().st_size == 0:
                print("▶ Converting audio...")
                run([
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", str(source),
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    str(audio)
                ])

            # Step 4: Run LeoTranscribe
            print("▶ Running LeoTranscribe...")
            run([sys.executable, str(Path(__file__).resolve().parent / "LeoTranscribe.py"), "--no-organize", str(audio)])

        # Step 5: Verify TXT
        if not transcript.is_file() or transcript.stat().st_size == 0:
            raise RuntimeError(
                f"Transcript missing or empty: {transcript}"
            )

        # Step 6: Move TXT to final destination
        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

        shutil.move(str(transcript), str(destination))

        # Step 7: Verify moved TXT
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("TXT verification failed")

        # Step 8: Delete this video's temporary files
        shutil.rmtree(work)

        print(f"✓ SAVED: {destination}")
        print("✓ Temporary audio deleted")

        return True

    except Exception as error:
        print(f"✗ FAILED: {title}")
        print(f"Reason: {error}")
        print(f"Recovery directory: {work}")
        return False


def main():
    if len(sys.argv) != 2:
        print('Usage: leoyt "YouTube URL"')
        sys.exit(1)

    url = sys.argv[1]

    for dependency in ("yt-dlp", "ffmpeg"):
        if not shutil.which(dependency):
            print(f"✗ Missing dependency: {dependency}")
            sys.exit(1)

    # Automatically restore folders if deleted or moved to SSD
    ensure_directory(OUTPUT)
    ensure_directory(CACHE)

    print("LeoYT 2.1")
    print("▶ Checking output directories...")
    print(f"✓ Output: {OUTPUT}")
    print(f"✓ Cache: {CACHE}")

    print("▶ Reading YouTube metadata...")
    metadata = get_metadata(url)

    entries = metadata.get("entries")

    if entries is not None:
        videos = [
            item for item in entries
            if item and item.get("id")
        ]

        if not videos:
            raise RuntimeError("Playlist contains no videos")

        playlist_name = safe_name(
            metadata.get("title") or "YouTube Playlist"
        )

        folder = ensure_directory(OUTPUT / playlist_name)

        total = len(videos)
        width = max(2, len(str(total)))
        completed = 0

        print(f"\nPlaylist: {playlist_name}")
        print(f"Videos: {total}")
        print(f"Destination: {folder}")

        for index, video in enumerate(videos, 1):
            video_id = video["id"]
            title = safe_name(video.get("title") or video_id)

            filename = f"{index:0{width}d}. {title}.txt"
            destination = folder / filename

            print(f"\n{'=' * 50}")
            print(f"[{index}/{total}] {title}")
            print(f"{'=' * 50}")

            if transcribe(video_id, title, destination):
                completed += 1

        print(f"\nFinished: {completed}/{total}")
        print(f"Saved to: {folder}")

        if completed != total:
            sys.exit(1)

    else:
        video_id = metadata.get("id")

        if not video_id:
            raise RuntimeError("Video ID missing")

        title = safe_name(metadata.get("title") or video_id)
        destination = OUTPUT / f"{title}.txt"

        print(f"\nSingle video: {title}")

        if not transcribe(video_id, title, destination):
            sys.exit(1)

        print("\n✓ Completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Recovery files preserved.")
        sys.exit(130)
    except Exception as error:
        print(f"\n✗ Error: {error}")
        sys.exit(1)
