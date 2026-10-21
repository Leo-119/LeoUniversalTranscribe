#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import subprocess


def load_audio(path):
    command = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def merge_intervals(intervals, gap=0.5):
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def subtract_intervals(speech, transcript):
    missing = []

    for start, end in speech:
        cursor = start

        for ts, te in transcript:
            if te <= cursor:
                continue

            if ts >= end:
                break

            if ts > cursor:
                missing.append([cursor, min(ts, end)])

            cursor = max(cursor, te)

            if cursor >= end:
                break

        if cursor < end:
            missing.append([cursor, end])

    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--min-gap", type=float, default=2.0)
    args = parser.parse_args()

    audio = Path(args.audio).expanduser()
    segments_path = audio.with_suffix(".segments.json")

    if not audio.exists():
        raise SystemExit(f"Audio not found: {audio}")

    if not segments_path.exists():
        raise SystemExit(f"Segments not found: {segments_path}")

    print("Loading audio...")
    waveform = load_audio(audio)

    print("Loading Silero VAD...")
    model, utils = torch.hub.load(
        "snakers4/silero-vad",
        "silero_vad",
        trust_repo=True
    )

    get_speech_timestamps = utils[0]

    speech = get_speech_timestamps(
        torch.from_numpy(waveform),
        model,
        sampling_rate=16000,
        return_seconds=True
    )

    speech_intervals = merge_intervals(
        [(x["start"], x["end"]) for x in speech],
        gap=0.5
    )

    raw_segments = json.loads(segments_path.read_text())

    transcript_intervals = merge_intervals(
        [
            (float(x["start"]), float(x["end"]))
            for x in raw_segments
            if x.get("text", "").strip()
        ],
        gap=0.15
    )

    missing = subtract_intervals(
        speech_intervals,
        transcript_intervals
    )

    missing = [
        [round(s, 2), round(e, 2)]
        for s, e in missing
        if e - s >= args.min_gap
    ]

    report = {
        "audio": str(audio),
        "speech_intervals": len(speech_intervals),
        "transcript_intervals": len(transcript_intervals),
        "potential_missing_regions": missing,
        "note": (
            "VAD coverage gaps are candidates, not proof of "
            "missing transcription."
        )
    }

    output = audio.with_suffix(".coverage.json")
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print()
    print("Coverage audit complete.")
    print(f"Potential missing regions: {len(missing)}")
    print(f"Report: {output}")

    for start, end in missing[:30]:
        print(f"  {start:.2f}s - {end:.2f}s")


if __name__ == "__main__":
    main()
