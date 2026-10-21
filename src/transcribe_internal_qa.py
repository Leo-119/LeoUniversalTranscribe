#!/usr/bin/env python3
"""Internal QA for MLX Whisper. No Ollama or external LLM."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SR = 16000
WHISPER = "mlx-community/whisper-large-v3-mlx"

PORTUGUESE_MARKERS = (
    "não", "então", "porquê", "saudável",
    "bagunça", "plásticas", "reciclagem",
    "caixas", "entendi", "oficina",
    "temos", "gostamos", "pensando",
)

def save_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def stamp(t):
    m, s = divmod(float(t), 60)
    return f"{int(m):02d}:{s:05.2f}"

def language_score(text):
    text = text.lower()
    return sum(
        bool(re.search(r"\b" + re.escape(w) + r"\b", text))
        for w in PORTUGUESE_MARKERS
    )

def suspicious_language(text):
    return language_score(text) >= 2

def repetition_issue(text):
    words = re.findall(r"\b[\w']+\b", text.lower())

    # Only flag unusually long repeated sequences.
    # Normal speech like "no, no, no" is allowed.
    if len(words) < 12:
        return False

    for size in range(3, 9):
        for i in range(len(words) - 2 * size + 1):
            if words[i:i+size] == words[i+size:i+2*size]:
                return True

    return False

def build_groups(indices, segments, max_gap=3.0):
    groups = []

    for idx in indices:
        s = segments[idx]

        if groups and s["start"] - groups[-1]["end"] <= max_gap:
            groups[-1]["end"] = max(
                groups[-1]["end"], s["end"]
            )
            groups[-1]["indices"].append(idx)
        else:
            groups.append({
                "start": s["start"],
                "end": s["end"],
                "indices": [idx],
            })

    return groups

def recheck_audio(audio, start, end):
    import numpy as np
    import mlx_whisper

    # Small context padding to avoid cutting off words.
    start = max(0.0, start - 1.5)
    end = end + 1.5

    duration = end - start

    command = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-ss", str(start),
        "-i", str(audio),
        "-t", str(duration),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(SR),
        "-"
    ]

    proc = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE
    )

    samples = np.frombuffer(
        proc.stdout, dtype=np.float32
    ).copy()

    if len(samples) == 0:
        raise RuntimeError("Empty audio window")

    result = mlx_whisper.transcribe(
        samples,
        path_or_hf_repo=WHISPER,
        language=None,
        task="transcribe",
        verbose=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
        logprob_threshold=-1.0,
    )

    parts = []
    confidences = []

    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()

        if text:
            parts.append(text)

        if seg.get("avg_logprob") is not None:
            confidences.append(
                float(seg["avg_logprob"])
            )

    return {
        "start": start,
        "end": end,
        "text": " ".join(parts),
        "avg_logprob": (
            sum(confidences) / len(confidences)
            if confidences else None
        ),
    }

def audit(audio):
    segments_path = audio.with_suffix(".segments.json")
    report_path = audio.with_suffix(".qa.json")
    review_path = audio.with_suffix(".qa.review.txt")

    if not segments_path.exists():
        raise SystemExit(
            f"Missing Whisper cache: {segments_path}"
        )

    segments = json.loads(
        segments_path.read_text(encoding="utf-8")
    )

    print("[Internal QA] Checking transcript...", flush=True)

    issues = []
    language_indices = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        start = float(seg["start"])
        end = float(seg["end"])

        if end <= start:
            issues.append({
                "type": "invalid_timestamp",
                "time": start,
                "text": text,
            })

        if suspicious_language(text):
            language_indices.append(i)

        if repetition_issue(text):
            issues.append({
                "type": "possible_repetition",
                "time": start,
                "text": text,
            })

    groups = build_groups(
        language_indices, segments
    )

    reviews = []

    for i, group in enumerate(groups, 1):
        start = group["start"]
        end = group["end"]

        original = " ".join(
            segments[j]["text"]
            for j in group["indices"]
        )

        print(
            f"[Internal QA] Rechecking "
            f"{stamp(start)}–{stamp(end)} "
            f"({i}/{len(groups)})",
            flush=True
        )

        try:
            candidate = recheck_audio(
                audio, start, end
            )

            reviews.append({
                "start": start,
                "end": end,
                "original": original,
                "candidate": candidate["text"],
                "candidate_avg_logprob":
                    candidate["avg_logprob"],
                "status": "needs_comparison",
            })

        except Exception as exc:
            reviews.append({
                "start": start,
                "end": end,
                "original": original,
                "status": "recheck_failed",
                "error": str(exc),
            })

    report = {
        "audio": str(audio),
        "engine": "Whisper Internal QA",
        "external_llm": False,
        "segments_checked": len(segments),
        "issues": issues,
        "language_rechecks": reviews,
        "original_transcript_preserved": True,
        "note": (
            "Language detection is heuristic. "
            "Candidate re-ASR is not automatically "
            "accepted as correct."
        ),
    }

    save_json(report_path, report)

    lines = [
        "WHISPER INTERNAL QA REPORT",
        "=" * 60,
        f"Audio: {audio.name}",
        f"Segments checked: {len(segments)}",
        f"Rule-based issues: {len(issues)}",
        f"Language regions rechecked: {len(reviews)}",
        "",
        "Original transcript has NOT been modified.",
        "",
    ]

    for item in issues:
        lines.extend([
            f"[{stamp(item['time'])}] "
            f"{item['type']}",
            f"  {item['text']}",
            "",
        ])

    for item in reviews:
        lines.extend([
            "=" * 60,
            f"RE-ASR [{stamp(item['start'])}"
            f"–{stamp(item['end'])}]",
            "",
            "ORIGINAL:",
            item["original"],
            "",
            "WHISPER SECOND PASS (ENGLISH):",
            item.get(
                "candidate",
                item.get("error", "Unavailable")
            ),
            "",
            f"Status: {item['status']}",
            "",
        ])

    review_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        f"[Internal QA] Checked {len(segments)} segments",
        flush=True
    )
    print(
        f"[Internal QA] Found {len(issues)} "
        f"rule-based issues",
        flush=True
    )
    print(
        f"[Internal QA] Rechecked {len(reviews)} "
        f"language regions",
        flush=True
    )
    print(f"[Internal QA] Report: {review_path}")
    print("[Internal QA] Original transcript preserved.")

def main():
    parser = argparse.ArgumentParser(
        description="Whisper + Internal QA"
    )

    parser.add_argument("audio")
    parser.add_argument(
        "--qa-only",
        action="store_true"
    )
    parser.add_argument(
        "--no-qa",
        action="store_true"
    )
    parser.add_argument(
        "--fresh",
        action="store_true"
    )
    parser.add_argument(
        "--stage",
        choices=["whisper", "diarization"]
    )

    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()

    if not audio.is_file():
        parser.error(f"Audio not found: {audio}")

    original = (
        Path(__file__).resolve().parent / "transcribe_before_grok.py"
    )

    if not original.is_file():
        parser.error(
            "Missing original transcription engine: "
            f"{original}"
        )

    if not args.qa_only:
        command = [
            sys.executable,
            str(original),
            str(audio),
        ]

        if args.fresh:
            command.append("--fresh")

        if args.stage:
            command.extend([
                "--stage", args.stage
            ])

        subprocess.run(command, check=True)

    if args.stage or args.no_qa:
        return

    audit(audio)

if __name__ == "__main__":
    main()
