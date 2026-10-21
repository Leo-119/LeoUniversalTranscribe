#!/usr/bin/env python3
"""
Unified Whisper transcription pipeline.

Features:
- Single-line progress display
- Original Whisper/VAD/pyannote pipeline
- Internal QA
- Silero VAD coverage audit
- Targeted Whisper re-ASR
- No Ollama dependency
- Original transcript preserved
"""

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path


HOME = Path.home()
SOURCE_DIR = Path(__file__).resolve().parent

ENGINE = SOURCE_DIR / "transcribe_before_grok.py"
INTERNAL_QA = SOURCE_DIR / "transcribe_internal_qa.py"
COVERAGE = SOURCE_DIR / "transcribe_coverage.py"

BAR_WIDTH = 28


def progress(label, fraction, detail=""):
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * BAR_WIDTH)

    bar = "█" * filled + "░" * (BAR_WIDTH - filled)

    line = (
        f"\r\033[2K{label:<15} "
        f"[{bar}] {fraction * 100:5.1f}%"
    )

    if detail:
        line += f"  {detail}"

    print(line, end="", flush=True)


def finish(label):
    progress(label, 1.0)


def run_quiet(command, label, logfile, parse_chunks=False):
    """
    Run subprocess while keeping its complete output in a log.
    Display only one progress line.
    """

    import pty

    master, slave = pty.openpty()

    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env=os.environ.copy(),
    )

    os.close(slave)

    last_chunk = 0
    total_chunks = 0
    recent = ""
    last_update = time.monotonic()

    progress(label, 0)

    with logfile.open("a", encoding="utf-8") as log:
        try:
            while True:
                ready, _, _ = select.select(
                    [master], [], [], 0.25
                )

                if ready:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break

                    if not data:
                        break

                    decoded = data.decode(
                        "utf-8", errors="replace"
                    )

                    log.write(decoded)
                    log.flush()

                    recent = (recent + decoded)[-4096:]

                    if parse_chunks:
                        matches = re.findall(
                            r"Whisper chunk\s+(\d+)/(\d+)",
                            recent
                        )

                        if matches:
                            current, total = matches[-1]
                            current = int(current)
                            total = int(total)

                            if total > 0:
                                last_chunk = current
                                total_chunks = total

                                progress(
                                    label,
                                    current / total,
                                    f"{current}/{total} chunks"
                                )

                    last_update = time.monotonic()

                if proc.poll() is not None and not ready:
                    break

                if time.monotonic() - last_update > 2:
                    if total_chunks:
                        progress(
                            label,
                            last_chunk / total_chunks,
                            f"{last_chunk}/{total_chunks} chunks"
                        )
                    else:
                        progress(label, 0, "working...")

                    last_update = time.monotonic()

        finally:
            os.close(master)

    returncode = proc.wait()

    if returncode:
        print()
        raise RuntimeError(
            f"{label} failed (exit {returncode}). "
            f"See log: {logfile}"
        )

    finish(label)


def load_coverage(audio):
    path = audio.with_suffix(".coverage.json")

    if not path.exists():
        return []

    report = json.loads(path.read_text(encoding="utf-8"))

    return report.get("potential_missing_regions", [])


def recheck_audio(audio, start, end):
    """
    Reuse the existing Internal QA Whisper re-ASR function.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "internal_qa_module",
        INTERNAL_QA
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.recheck_audio(audio, start, end)


def targeted_reasr(audio, regions):
    output = audio.with_suffix(".coverage.review.json")
    text_output = audio.with_suffix(".coverage.review.txt")

    results = []

    for index, region in enumerate(regions, 1):
        start, end = map(float, region)

        progress(
            "Re-ASR",
            (index - 1) / max(len(regions), 1),
            f"{index}/{len(regions)}"
        )

        try:
            candidate = recheck_audio(audio, start, end)

            results.append({
                "start": start,
                "end": end,
                "candidate": candidate.get("text", ""),
                "avg_logprob": candidate.get("avg_logprob"),
                "status": "needs_review",
            })

        except Exception as exc:
            results.append({
                "start": start,
                "end": end,
                "status": "failed",
                "error": str(exc),
            })

    if regions:
        finish("Re-ASR")

    output.write_text(
        json.dumps(
            {
                "audio": str(audio),
                "results": results,
                "original_preserved": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8"
    )

    lines = [
        "COVERAGE RE-ASR REVIEW",
        "=" * 60,
        f"Audio: {audio.name}",
        "",
        "Original transcript has NOT been modified.",
        "",
    ]

    for item in results:
        lines.extend([
            f"Region: {item['start']:.2f}s"
            f"–{item['end']:.2f}s",
            f"Status: {item['status']}",
            "",
            item.get(
                "candidate",
                item.get("error", "")
            ),
            "",
            "-" * 60,
            "",
        ])

    text_output.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return text_output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("audio")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["whisper", "diarization"]
    )

    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()

    if not audio.is_file():
        parser.error(f"Audio not found: {audio}")

    logfile = audio.with_suffix(".pipeline.log")

    logfile.write_text(
        "UNIFIED TRANSCRIPTION PIPELINE\n",
        encoding="utf-8"
    )

    if not args.qa_only and not args.coverage_only:
        if not ENGINE.exists():
            parser.error(f"Missing engine: {ENGINE}")

        command = [
            sys.executable,
            str(ENGINE),
            str(audio),
        ]

        if args.fresh:
            command.append("--fresh")

        if args.stage:
            command.extend(["--stage", args.stage])

        run_quiet(
            command,
            "Whisper",
            logfile,
            parse_chunks=True,
        )

    if args.stage:
        print("Requested stage completed.")
        return

    if not args.coverage_only and not args.no_qa:
        if not INTERNAL_QA.exists():
            parser.error(
                f"Missing Internal QA: {INTERNAL_QA}"
            )

        run_quiet(
            [
                sys.executable,
                str(INTERNAL_QA),
                str(audio),
                "--qa-only",
            ],
            "Internal QA",
            logfile,
        )

    if not args.no_coverage:
        if not COVERAGE.exists():
            parser.error(
                f"Missing Coverage Audit: {COVERAGE}"
            )

        run_quiet(
            [
                sys.executable,
                str(COVERAGE),
                str(audio),
            ],
            "Coverage",
            logfile,
        )

        regions = load_coverage(audio)

        # Gap count is available in .coverage.json.

        # Calibration handles re-ASR once; avoid duplicate model calls.

    if not args.no_coverage:
        calibration = SOURCE_DIR / "transcribe_calibration.py"

        if not calibration.exists():
            raise RuntimeError(
                f"Calibration module missing: {calibration}"
            )

        run_quiet(
            [
                sys.executable,
                str(calibration),
                str(audio),
            ],
            "Calibration",
            logfile,
        )


    print("\nPipeline complete.")
    print(f"Full log: {logfile}")
    print("Original transcript preserved.")


if __name__ == "__main__":
    main()
