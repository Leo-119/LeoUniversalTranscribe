#!/usr/bin/env python3
"""Closed-loop transcription calibration, candidate-only."""

import argparse
import contextlib
import difflib
import importlib.util
import io
import json
import re
import sys
import os
import statistics
from datetime import datetime
from pathlib import Path

HOME = Path.home()
QA_MODULE = Path(__file__).resolve().parent / "transcribe_internal_qa.py"


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(
        r"[^\w\s]", " ", text.lower()
    )).strip()


def similarity(a, b):
    a, b = normalize(a), normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def load_segments(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = data.get("segments", [])

    if not isinstance(data, list):
        raise ValueError("Unexpected segments JSON format")

    result = []

    for item in data:
        if not isinstance(item, dict):
            continue

        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, ValueError, TypeError):
            continue

        if end <= start:
            continue

        result.append({
            "start": start,
            "end": end,
            "text": str(item.get("text", "")),
        })

    return sorted(result, key=lambda x: x["start"])


def load_coverage(path):
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    regions = []

    for pair in data.get("potential_missing_regions", []):
        try:
            start, end = map(float, pair)
        except (ValueError, TypeError):
            continue

        if end > start:
            regions.append({
                "start": start,
                "end": end,
                "reason": "speech_coverage_gap",
                "score": 0.85,
            })

    return regions


def detect_anomalies(segments):
    regions = []

    for seg in segments:
        text = seg["text"].strip()

        if not text:
            continue

        words = normalize(text).split()

        if len(words) >= 8:
            repeated = False

            for width in range(2, min(9, len(words) // 2 + 1)):
                for i in range(len(words) - 2 * width + 1):
                    if (
                        words[i:i + width]
                        == words[i + width:i + 2 * width]
                    ):
                        repeated = True
                        break

                if repeated:
                    break

            if repeated:
                regions.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "reason": "possible_repetition",
                    "score": 0.55,
                })

        # Multilingual recordings may legitimately contain CJK,
        # Cyrillic and other scripts.
        # Do not classify a segment as anomalous solely by script.
        # Other anomaly detectors remain active.

    return regions


def merge_regions(regions, max_gap=1.5):
    if not regions:
        return []

    regions = sorted(
        regions,
        key=lambda r: (r["start"], r["end"])
    )

    merged = [dict(regions[0])]
    merged[0]["reasons"] = [merged[0].pop("reason")]

    for region in regions[1:]:
        last = merged[-1]

        if region["start"] <= last["end"] + max_gap:
            last["end"] = max(last["end"], region["end"])
            last["score"] = max(last["score"], region["score"])
            if region["reason"] not in last["reasons"]:
                last["reasons"].append(region["reason"])
        else:
            item = dict(region)
            item["reasons"] = [item.pop("reason")]
            merged.append(item)

    return merged


def nearby_text(segments, start, end, margin=2.0):
    selected = [
        s["text"].strip()
        for s in segments
        if s["end"] >= start - margin
        and s["start"] <= end + margin
        and s["text"].strip()
    ]

    return " ".join(selected)


def load_rechecker():
    if not QA_MODULE.exists():
        raise FileNotFoundError(
            f"Internal QA module missing: {QA_MODULE}"
        )

    spec = importlib.util.spec_from_file_location(
        "calibration_internal_qa",
        QA_MODULE
    )

    module = importlib.util.module_from_spec(spec)

    # Suppress model initialization chatter.
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)

    fn = getattr(module, "recheck_audio", None)

    if not callable(fn):
        raise RuntimeError(
            "Internal QA has no callable recheck_audio(). "
            "Original files remain unchanged."
        )

    return fn


def run_recheck(fn, audio, start, end, log):
    # Native MLX/tqdm output bypasses redirect_stdout; redirect file descriptors.
    with log.open("a", encoding="utf-8") as stream:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(stream.fileno(), 1); os.dup2(stream.fileno(), 2)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = fn(audio, start, end)
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved_out, 1); os.dup2(saved_err, 2)
            os.close(saved_out); os.close(saved_err)

    if isinstance(result, dict):
        return str(result.get("text", "")).strip()

    if isinstance(result, str):
        return result.strip()

    return ""


def financial_flags(text):
    """Flag potentially important numeric content, not verify it."""
    pattern = (
        r"\b(?:PV|FV|PMT|P/Y|C/Y|I/Y)\b"
        r"|\$\s*\d"
        r"|\b\d+(?:\.\d+)?\s*%"
    )

    return re.findall(pattern, text, flags=re.I)


def progress(current, total, label):
    width = 28
    fraction = current / max(total, 1)
    filled = round(fraction * width)

    bar = "█" * filled + "░" * (width - filled)

    print(
        f"\r\033[2K{label} [{bar}] "
        f"{fraction * 100:5.1f}% ({current}/{total})",
        end="",
        flush=True,
    )


def score_candidate(text, original, previous, margin, baseline_margin):
    """Conservative ranking, not an accuracy probability."""
    if not text.strip():
        return -1.0
    # Context expansion creates extra words; do not reward longer outputs.
    length_ratio = len(normalize(text).split()) / max(1, len(normalize(original).split()))
    original_agreement = similarity(text, original)
    prior_agreement = max([similarity(text, p) for p in previous if p] or [0.0])
    length_penalty = min(0.4, max(0, length_ratio - 1.6) * 0.10)
    window_penalty = 0.015 * max(0, margin - baseline_margin)
    return round(0.60 * original_agreement + 0.40 * prior_agreement - length_penalty - window_penalty, 4)


def calibrate(audio, max_regions=30, max_passes=3):
    segments_path = audio.with_suffix(".segments.json")
    coverage_path = audio.with_suffix(".coverage.json")

    if not segments_path.exists():
        raise FileNotFoundError(segments_path)

    segments = load_segments(segments_path)

    regions = merge_regions(
        load_coverage(coverage_path)
        + detect_anomalies(segments)
    )

    regions = sorted(
        regions,
        key=lambda x: x["score"],
        reverse=True
    )

    skipped = max(0, len(regions) - max_regions)
    regions = regions[:max_regions]

    output_json = audio.with_suffix(".calibration.json")
    output_txt = audio.with_suffix(".calibration.review.txt")
    logfile = audio.with_suffix(".calibration.log")

    logfile.write_text(
        "CLOSED-LOOP CALIBRATION LOG\n",
        encoding="utf-8"
    )

    results = []

    if regions:
        recheck = load_rechecker()
    else:
        recheck = None

    for index, region in enumerate(regions, 1):
        start = region["start"]
        end = region["end"]

        original = nearby_text(
            segments, start, end, margin=0.5
        )

        candidates = []
        history = []
        baseline_margin = 1.5

        # Increasing context windows: P -> I -> D-inspired
        # adaptation. These are not literal PID gains.
        margins = [1.5, 3.0, 5.0][:max_passes]

        for attempt, margin in enumerate(margins, 1):
            left = max(0.0, start - margin)
            right = end + margin

            try:
                candidate = run_recheck(
                    recheck, audio, left, right, logfile
                )
                error = None
            except Exception as exc:
                candidate = ""
                error = str(exc)

            rank_score = score_candidate(candidate, original, candidates, margin, baseline_margin)
            candidates.append(candidate)

            previous = (
                candidates[-2]
                if len(candidates) >= 2
                else original
            )

            agreement = similarity(candidate, previous)

            history.append({
                "pass": attempt,
                "window": [round(left, 2), round(right, 2)],
                "text": candidate,
                "similarity_to_previous": round(agreement, 4),
                "error": error,
                "rank_score": rank_score,
            })

            progress(
                index - 1 + attempt / len(margins),
                len(regions),
                "Calibration"
            )

            # Two nonempty similar candidates suggest stability.
            # Not proof of correctness.
            if (
                attempt >= 2
                and candidate
                and previous
                and agreement >= 0.92
            ):
                break

        # Never assume the last/largest window is best. The heuristic
        # cannot establish acoustic correctness or align individual words.
        valid_history = [h for h in history if h["text"] and not h["error"]]
        chosen_record = max(valid_history, key=lambda h: h["rank_score"], default=None)
        chosen = chosen_record["text"] if chosen_record else ""
        selected_pass = chosen_record["pass"] if chosen_record else None
        if not valid_history:
            status = "NO_RELIABLE_CANDIDATE"
        elif original and similarity(chosen, original) >= 0.92:
            status = "LIKELY_ALREADY_TRANSCRIBED"
        elif len(valid_history) >= 2 and max(
            similarity(a["text"], b["text"])
            for i, a in enumerate(valid_history)
            for b in valid_history[i+1:]
        ) >= 0.92:
            status = "STABLE_CANDIDATE_REVIEW"
        else:
            status = "UNRESOLVED_REVIEW"

        # Crucial: candidate may include speech outside the target window.
        # Do not call it recovered speech or insert into original transcript.
        original_similarity = similarity(original, chosen)

        results.append({
            "start": start,
            "end": end,
            "reasons": region["reasons"],
            "risk_score": region["score"],
            "original_context": original,
            "candidate": chosen,
            "selected_pass": selected_pass,
            "selection_basis": "heuristic only; no word-level acoustic alignment",
            "status": status,
            "financial_tokens_original": financial_flags(original),
            "financial_tokens_candidate": financial_flags(chosen),
            "history": history,
        })

        progress(index, len(regions), "Calibration")

    if regions:
        print()
    else:
        print("Calibration: no regions triggered by current rules.")

    report = {
        "version": "3.0",
        "created": datetime.now().isoformat(),
        "audio": str(audio),
        "segments_checked": len(segments),
        "regions_checked": len(results),
        "regions_skipped_due_to_limit": skipped,
        "original_transcript_modified": False,
        "limitations": [
            "Repeated Whisper outputs are correlated evidence.",
            "Stable text is not proof of accurate transcription.",
            "Numeric tokens are flagged, not mathematically verified.",
            "Language detection is script-based, not full language ID.",
            "No automatic corrections are applied.",
            "Expanded windows can include unrelated adjacent speech.",
            "Ranking scores are not confidence or accuracy probabilities.",
        ],
        "results": results,
    }

    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    lines = [
        "WHISPER CLOSED-LOOP CALIBRATION V3",
        "=" * 65,
        f"Audio: {audio.name}",
        f"Segments: {len(segments)}",
        f"Regions checked: {len(results)}",
        f"Regions skipped: {skipped}",
        "Original transcript preserved.",
        "",
    ]

    for item in results:
        lines.extend([
            f"TIME: {item['start']:.2f}–{item['end']:.2f}",
            f"REASONS: {', '.join(item['reasons'])}",
            f"STATUS: {item['status']}",
            f"SELECTED PASS: {item['selected_pass']}",
            "",
            "ORIGINAL:",
            item["original_context"] or "[empty]",
            "",
        ])

        for attempt in item["history"]:
            lines.extend([
                f"PASS {attempt['pass']} (rank={attempt['rank_score']}):",
                attempt["text"] or "[empty]",
                "",
            ])

        lines.extend([
            "SELECTED REVIEW CANDIDATE (NOT AUTO-CORRECTED):",
            item["candidate"] or "[empty]",
            "",
            "-" * 65,
            "",
        ])

    output_txt.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"Calibration regions: {len(results)}")
    print(f"Review: {output_txt}")
    print(f"JSON: {output_json}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--max-regions", type=int, default=30)
    parser.add_argument("--max-passes", type=int, default=3)

    args = parser.parse_args()

    if args.max_regions < 1:
        parser.error("--max-regions must be positive")

    if not 1 <= args.max_passes <= 3:
        parser.error("--max-passes must be 1, 2 or 3")

    audio = Path(args.audio).expanduser().resolve()

    if not audio.is_file():
        parser.error(f"Audio not found: {audio}")

    calibrate(
        audio,
        max_regions=args.max_regions,
        max_passes=args.max_passes,
    )


if __name__ == "__main__":
    main()
