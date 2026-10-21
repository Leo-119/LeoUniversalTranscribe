#!/usr/bin/env python3

import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

def show_cli_info():
    import sys

    if len(sys.argv) != 2:
        return False

    option = sys.argv[1]

    if option in ("-h", "--help"):
        print("""LeoUniversalTranscribe

Usage:
  python3 src/LeoUniversalTranscribe.py <input> [options]

Input:
  Local audio/video file
  YouTube video URL

Local options:
  --fresh
  --no-organize
  --course <profile>
  --verify-regions <number>

Other:
  -h, --help
  --version
""")
        return True

    if option == "--version":
        print("LeoUniversalTranscribe 1.0.0-dev")
        return True

    return False


if __name__ == "__main__" and show_cli_info():
    raise SystemExit(0)



YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

MEDIA_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".flac",
    ".aac", ".ogg", ".opus", ".aiff",
    ".mp4", ".mov", ".mkv", ".webm", ".qta",
}

def is_youtube_url(value):
    parsed = urlparse(value)
    return (
        parsed.scheme in ("http", "https")
        and (parsed.hostname or "").lower() in YOUTUBE_HOSTS
    )

def main():
    if len(sys.argv) < 2:
        print(
            "LeoUniversalTranscribe\n"
            "\n"
            "Usage:\n"
            "  LeoUniversalTranscribe <media file>\n"
            "  LeoUniversalTranscribe <YouTube URL>\n"
            "\n"
            "Options for local media:\n"
            "  --fresh\n"
            "  --course <profile>\n"
            "  --verify-regions <number>\n"
            "  --no-organize"
        )
        return 1

    source = sys.argv[1]
    extra = sys.argv[2:]

    if is_youtube_url(source):
        if extra:
            print(
                "Error: Additional arguments are not "
                "supported for YouTube inputs.",
                file=sys.stderr
            )
            return 2

        command = [sys.executable, str(Path(__file__).resolve().parent / "LeoYT.py"), source]
        input_type = "YouTube"

    else:
        path = Path(source).expanduser().resolve()

        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 2

        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            print(
                f"Unsupported file type: {path.suffix}",
                file=sys.stderr
            )
            return 2

        command = [sys.executable, str(Path(__file__).resolve().parent / "LeoTranscribe.py"), str(path), *extra]
        input_type = "Local Media"

        if path.suffix.lower() == ".qta":
            if not shutil.which("ffmpeg"):
                print("Missing dependency: ffmpeg", file=sys.stderr)
                return 2

            with tempfile.TemporaryDirectory(prefix="leo_qta_") as tmp:
                converted = Path(tmp) / (path.stem + ".m4a")

                result = subprocess.run([
                    "ffmpeg", "-nostdin", "-hide_banner",
                    "-loglevel", "error", "-n",
                    "-i", str(path),
                    "-map", "0:a:0",
                    "-c:a", "copy",
                    str(converted)
                ])

                if result.returncode != 0:
                    print("QTA conversion failed.", file=sys.stderr)
                    return result.returncode

                result = subprocess.run([
                    sys.executable, str(Path(__file__).resolve().parent / "LeoTranscribe.py"), str(converted), *extra
                ])

                transcript = converted.with_suffix(".txt")

                if transcript.is_file():
                    destination = path.with_suffix(".txt")

                    if destination.exists():
                        print(
                            f"Transcript already exists: {destination}",
                            file=sys.stderr
                        )
                        print(
                            f"Recovered TXT remains at: {transcript}",
                            file=sys.stderr
                        )
                        return 3

                    shutil.copy2(transcript, destination)
                    print(
                        f"Final TXT saved: {destination}",
                        flush=True
                    )
                else:
                    print(
                        "Warning: No TXT found beside temporary audio.",
                        file=sys.stderr
                    )

                return result.returncode

    if not Path(command[1]).is_file():
        print(
            f"Missing dependency: {command[0]}",
            file=sys.stderr
        )
        return 2

    print("=" * 55, flush=True)
    print("LeoUniversalTranscribe", flush=True)
    print(f"Input type: {input_type}", flush=True)
    print("=" * 55, flush=True)

    try:
        return subprocess.run(command).returncode
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    sys.exit(main())
