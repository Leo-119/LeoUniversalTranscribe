import os
#!/usr/bin/env python3

import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("LEOTRANSCRIBE_OUTPUT_DIR", str(Path.home() / "Desktop" / "Transcripts"))).expanduser()
MODEL = "mistral-open:latest"

CATEGORIES = {
    "school": "School Recordings",
    "recording": "Recordings",
    "youtube": "YouTube",
    "lecture": "Bitch-ass Lecture",
    "unsorted": "Unsorted",
}

def classify(path):
    text = path.read_text(encoding="utf-8")

    # Remove the heuristic navigation index and focus on actual transcript.
    if "FULL TRANSCRIPT\n" in text:
        text = text.split("FULL TRANSCRIPT\n", 1)[1]

    # Sample beginning, middle, and end for long recordings.
    n = len(text)
    sample_size = 5000

    if n > sample_size * 3:
        sample = (
            text[:sample_size]
            + "\n...\n"
            + text[n // 2:n // 2 + sample_size]
            + "\n...\n"
            + text[-sample_size:]
        )
    else:
        sample = text

    prompt = f"""
Classify the following audio transcript into EXACTLY ONE category.

Categories:

school:
Formal school or college classes, coursework, professors,
assignments, quizzes, exams, classroom discussions.

recording:
Personal conversations, interviews, meetings, casual recordings.

youtube:
YouTube videos, online creator content, gaming commentary,
video essays and online entertainment.

lecture:
Non-school lectures, public talks, seminars, presentations.

unsorted:
Insufficient evidence to determine the category.

IMPORTANT:
- Classify based on transcript CONTENT, not filename.
- College lectures belong to school, NOT lecture.
- Do not infer YouTube merely because someone mentions YouTube.
- If uncertain, return unsorted.
- Return ONLY one lowercase category identifier.

Transcript:
{sample}
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 30,
        },
        "keep_alive": "0",
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.load(response)

    answer = result.get("response", "").strip().lower()
    match = re.fullmatch(
        r'["\']?(school|recording|youtube|lecture|unsorted)["\']?\.?',
        answer,
    )

    return match.group(1) if match else "unsorted"


def organize(path):
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        category = classify(path)
    except Exception as exc:
        print(f"AI classification unavailable: {exc}")
        category = "unsorted"

    destination_dir = ROOT / CATEGORIES[category]
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / path.name

    # Never overwrite an existing transcript.
    counter = 1
    while destination.exists():
        if destination.resolve() == path:
            print(f"Already organized: {destination}")
            return destination

        destination = destination_dir / (
            f"{path.stem} ({counter}){path.suffix}"
        )
        counter += 1

    shutil.move(str(path), str(destination))

    print(f"AI category: {category}")
    print(f"Organized TXT: {destination}")

    return destination


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: TranscriptOrganizer.py <transcript.txt>")

    organize(sys.argv[1])
