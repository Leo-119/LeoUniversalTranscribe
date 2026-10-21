#!/usr/bin/env python3
"""GPU-only speech model inference; separate subprocesses for MLX and pyannote MPS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SR = 16000
WHISPER = 'mlx-community/whisper-large-v3-mlx'
PYANNOTE = 'pyannote/speaker-diarization-community-1'


def decode(path):
    import numpy as np
    proc = subprocess.run([
        'ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '1', '-ar', str(SR), '-'
    ], check=True, stdout=subprocess.PIPE)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def write_json(path, data):
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)



def whisper_stage(audio, segments_path):
    import mlx_whisper
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    print('[1/3] Silero VAD + MLX Whisper Large-v3 — GPU', flush=True)

    samples = decode(audio)
    duration = len(samples) / SR
    print(f'Audio length: {duration / 60:.1f} min', flush=True)

    # Silero VAD uses lightweight CPU inference.
    model = load_silero_vad()

    timestamps = get_speech_timestamps(
        torch.from_numpy(samples),
        model,
        sampling_rate=SR,
        threshold=0.4,
        min_speech_duration_ms=200,
        min_silence_duration_ms=500,
        speech_pad_ms=300,
        return_seconds=True,
    )

    print(f'VAD detected {len(timestamps)} speech intervals', flush=True)

    # Merge nearby speech intervals to preserve conversational context.
    # Split long regions to keep Whisper input manageable.
    regions = []
    MAX_LENGTH = 25.0
    MAX_GAP = 2.0

    for item in timestamps:
        start = float(item['start'])
        end = float(item['end'])

        if regions and start - regions[-1][1] <= MAX_GAP:
            if end - regions[-1][0] <= MAX_LENGTH:
                regions[-1][1] = max(regions[-1][1], end)
                continue

        regions.append([start, end])

    # Split any unusually long continuous regions.
    chunks = []

    for start, end in regions:
        pos = start
        while pos < end:
            chunk_end = min(pos + MAX_LENGTH, end)
            chunks.append((pos, chunk_end))
            pos = chunk_end

    print(f'Processing {len(chunks)} speech chunks', flush=True)

    segments = []

    for index, (start, end) in enumerate(chunks, 1):
        a = max(0, int(start * SR))
        b = min(len(samples), int(end * SR))

        if b <= a:
            continue

        chunk = samples[a:b]

        result = mlx_whisper.transcribe(
            chunk,
            path_or_hf_repo=WHISPER,
            language=None,
            task='transcribe',
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.0,
            logprob_threshold=-1.0,

                     initial_prompt=os.environ.get('LEOTRANSCRIBE_INITIAL_PROMPT') or None,
                 )

        for s in result['segments']:
            text = s['text'].strip()

            if not text:
                continue

            # Restore timestamps to the original audio timeline.
            absolute_start = start + float(s['start'])
            absolute_end = start + float(s['end'])

            absolute_start = max(start, min(absolute_start, end))
            absolute_end = max(absolute_start, min(absolute_end, end))

            segments.append({
                'start': absolute_start,
                'end': absolute_end,
                'text': text,
            })

        print(
            f'Whisper chunk {index}/{len(chunks)} '
            f'({start:.1f}s–{end:.1f}s)',
            flush=True,
        )

    segments.sort(key=lambda s: (s['start'], s['end']))

    write_json(segments_path, segments)

    print(
        f'Saved {len(segments)} segments to {segments_path}',
        flush=True,
    )


def diarization_stage(audio, speakers_path):
    import torch
    from pyannote.audio import Pipeline
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; refusing CPU fallback.')
    print('[2/3] pyannote Community-1 — MPS GPU', flush=True)
    pipeline = Pipeline.from_pretrained(PYANNOTE)
    if pipeline is None:
        raise RuntimeError('pyannote model failed to load.')
    pipeline.to(torch.device('mps'))
    waveform = torch.from_numpy(decode(audio)).unsqueeze(0)
    with torch.inference_mode():
        result = pipeline({'waveform': waveform, 'sample_rate': SR})
    annotation = getattr(result, 'speaker_diarization', result)
    turns = [{'start': float(turn.start), 'end': float(turn.end), 'speaker': str(label)}
             for turn, _, label in annotation.itertracks(yield_label=True)]
    write_json(speakers_path, turns)
    print(f'Saved {len(turns)} speaker turns to {speakers_path}', flush=True)


def stamp(t):
    minutes, seconds = divmod(float(t), 60)
    return f'{int(minutes):02d}:{seconds:05.2f}'



def merge(segments_path, speakers_path, output):
    from collections import defaultdict
    from datetime import datetime

    segments = json.loads(segments_path.read_text(encoding='utf-8'))
    turns = json.loads(speakers_path.read_text(encoding='utf-8'))

    ids = sorted({t['speaker'] for t in turns})
    labels = {sid: f'Speaker {i + 1}' for i, sid in enumerate(ids)}

    def assign_speaker(segment):
        scores = defaultdict(float)

        for turn in turns:
            overlap = max(
                0.0,
                min(segment['end'], turn['end'])
                - max(segment['start'], turn['start'])
            )

            if overlap > 0:
                scores[turn['speaker']] += overlap

        # Rule 1: Prefer actual temporal overlap.
        if scores:
            return labels[max(scores, key=scores.get)]

        # Rule 2: Find the nearest diarization interval.
        midpoint = (segment['start'] + segment['end']) / 2

        nearest_speaker = None
        nearest_distance = float('inf')

        for turn in turns:
            if midpoint < turn['start']:
                distance = turn['start'] - midpoint
            elif midpoint > turn['end']:
                distance = midpoint - turn['end']
            else:
                distance = 0.0

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_speaker = turn['speaker']

        # Rule 3: Do not infer across long gaps.
        if nearest_distance <= 2.0:
            return labels[nearest_speaker]

        return 'Unknown'

    lines = []

    for segment in segments:
        speaker = assign_speaker(segment)

        lines.append(
            f"[{stamp(segment['start'])} --> "
            f"{stamp(segment['end'])}] "
            f"{speaker}: {segment['text']}"
        )

    fd, temp = tempfile.mkstemp(
        prefix=output.name + '.',
        suffix='.tmp',
        dir=output.parent
    )

    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

        if output.exists():
            backup = output.with_name(output.stem + '.old.txt')

            if backup.exists():
                backup = output.with_name(
                    output.stem + '.old-' +
                    datetime.now().strftime('%Y%m%d-%H%M%S') +
                    '.txt'
                )

            os.replace(output, backup)
            print(f'Previous TXT backed up: {backup}', flush=True)

        os.replace(temp, output)

    finally:
        if os.path.exists(temp):
            os.unlink(temp)

    print(f'[3/3] Done: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='MLX Whisper + pyannote MPS (separate processes)')
    parser.add_argument('audio')
    parser.add_argument('--fresh', action='store_true', help='Recompute both stages')
    parser.add_argument('--stage', choices=['whisper', 'diarization'])
    args = parser.parse_args()
    audio = Path(args.audio).expanduser().resolve()
    if not audio.is_file():
        parser.error(f'Audio file not found: {audio}')
    segments = audio.with_suffix('.segments.json')
    speakers = audio.with_suffix('.speakers.json')
    output = audio.with_suffix('.txt')
    if args.stage == 'whisper':
        whisper_stage(audio, segments)
        return
    if args.stage == 'diarization':
        diarization_stage(audio, speakers)
        return
    if args.fresh:
        for p in (segments, speakers):
            p.unlink(missing_ok=True)
    for stage, path in [('whisper', segments), ('diarization', speakers)]:
        if path.exists():
            print(f'Reusing saved result: {path}', flush=True)
            continue
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(audio), '--stage', stage], check=True)
    merge(segments, speakers, output)


if __name__ == '__main__':
    main()
