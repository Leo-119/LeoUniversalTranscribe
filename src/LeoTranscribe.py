#!/usr/bin/env python3
"""LeoTranscribe V4: isolated, verify-before-delete, single TXT publication."""
import argparse
import re
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import math
import unicodedata
from pathlib import Path

HOME = Path.home()
ROOT = Path(os.environ.get('LEOTRANSCRIBE_DATA_DIR', str(HOME / 'Library' / 'Application Support' / 'LeoTranscribe'))).expanduser()
JOBS = ROOT / 'Jobs'
ENGINE = Path(__file__).resolve().parent / 'engine.py'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src, dest):
    """Use APFS clone when possible; otherwise normal copy."""
    if sys.platform == 'darwin':
        result = subprocess.run(['cp', '-c', str(src), str(dest)], capture_output=True)
        if result.returncode == 0:
            return
    shutil.copy2(src, dest)


def verify(working_audio, original_audio, audit_only):
    """Structural checks; cannot establish audio-text accuracy."""
    if sha256(working_audio) != sha256(original_audio):
        raise RuntimeError('Audio integrity check failed')
    segments = working_audio.with_suffix('.segments.json')
    if not segments.is_file():
        raise RuntimeError('Missing Whisper segments; cannot verify')
    data = json.loads(segments.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('segments', [])
    if not isinstance(data, list) or not data:
        raise RuntimeError('Empty or malformed Whisper segments')
    for i, seg in enumerate(data):
        if not isinstance(seg, dict):
            raise RuntimeError(f'Malformed segment #{i}')
        try:
            start, end = float(seg['start']), float(seg['end'])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f'Malformed timestamps in segment #{i}') from exc
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < start:
            raise RuntimeError(f'Invalid timestamp range in segment #{i}: {start}–{end}')
        # Zero-duration segments may contain important text. Retain them and flag
        # their timing as unverified rather than silently deleting their words.

    txt = working_audio.with_suffix('.txt')
    if not txt.is_file() or not txt.read_text(encoding='utf-8').strip():
        raise RuntimeError('Missing or empty final transcript')
    cal = working_audio.with_suffix('.calibration.json')
    if not cal.is_file():
        raise RuntimeError('Missing calibration report; cannot finalize')
    report = json.loads(cal.read_text(encoding='utf-8'))
    return txt, report, len(data)


def extract_review(report):
    """Audit existing multi-pass Whisper calibration before deleting intermediates.

    The V3 calibration report contains actual repeat recognitions (history).
    This compares them; it never treats consensus as ground truth.
    """
    items = report.get('results', report.get('regions', []))
    if not isinstance(items, list):
        raise RuntimeError('Unexpected calibration report structure')
    lines = []
    checked = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get('status', 'UNKNOWN'))
        if status == 'LIKELY_ALREADY_TRANSCRIBED':
            continue
        start, end = item.get('start', '?'), item.get('end', '?')
        reason = item.get('reason', item.get('reasons', ''))
        history = item.get('history', [])
        candidates = [str(h.get('text', '')).strip() for h in history
                      if isinstance(h, dict) and h.get('text') and not h.get('error')]
        checked += len(candidates)
        summary = ''
        if len(candidates) >= 2:
            similarities = [difflib.SequenceMatcher(None,
                            candidates[i-1].lower(), candidates[i].lower()).ratio()
                            for i in range(1, len(candidates))]
            last_similarity = similarities[-1]
            summary = f'; Whisper passes: {len(candidates)}, final-pair similarity: {last_similarity:.0%}'
            if last_similarity < .85:
                summary += ' [DISAGREEMENT]'
            # Detect unstable numerical tokens across recognition passes.
            numbers = [set(re.findall(r'(?<![\w])[$]?-?\d[\d,.]*%?', t))
                       for t in candidates]
            if any(nums != numbers[-1] for nums in numbers[:-1]):
                summary += ' [NUMERIC DISAGREEMENT — CHECK AUDIO]'
        elif len(candidates) == 1:
            summary = '; only one successful Whisper pass'
        else:
            summary = '; no successful repeat Whisper passes available'
        candidate = str(item.get('candidate', '')).strip()
        # Keep the useful candidate before deleting the detailed report.
        if len(candidate) > 260:
            candidate = candidate[:257] + '...'
        lines.append(f'[{start}s–{end}s] {status}; reason: {reason}{summary}')
        if candidate:
            lines.append(f'  Whisper candidate (UNVERIFIED): {candidate}')
    skipped = report.get('regions_skipped_due_to_limit', 0)
    if skipped:
        lines.append(f'[Calibration skipped {skipped} lower-priority regions; not verified]')
    return lines, checked


# Domain prompts are recognition hints, not model weight changes.
PROFILES = {
    'math': ('Financial mathematics lecture. Time value of money, present value PV, future value FV, '
             'payment PMT, interest I/Y, payments per year P/Y, compounding per year C/Y, '
             'BA II Plus, BGN mode, END mode, annuity due, ordinary annuity, amortization, '
             'quarterly, semiannual, compounded monthly.'),
    'project': ('Project management lecture. Work breakdown structure WBS, critical path method CPM, '
                'Gantt chart, scope, stakeholders, milestones, dependencies, risk register, Agile, waterfall.'),
    'marketing': ('Marketing and consumer behaviour lecture. Segmentation, targeting, positioning, '
                  'consumer decision journey, brand equity, perception, attitudes, motivation, marketing mix.'),
    'supply': ('Supply chain management lecture. Procurement, logistics, inventory, lead time, '
               'economic order quantity EOQ, just-in-time JIT, safety stock, demand forecasting.'),
    'computer': ('Business computer applications lecture. Microsoft Project, Microsoft Access, '
                 'relational database, primary key, query, Gantt chart, task dependencies, resources.'),
    'general': '',
}

def detect_course(audio, override):
    if override != 'auto':
        return override
    name = audio.stem.lower()
    for label, terms in {
        'math': ('math', 'finance', 'interest', 'annuit', 'hlinka'),
        'project': ('project management', 'bus1040'),
        'marketing': ('marketing', 'consumer', 'mark2054'),
        'supply': ('supply chain', 'logistics'),
        'computer': ('computer app', 'comp1115', 'microsoft access'),
    }.items():
        if any(term in name for term in terms):
            return label
    return 'general'

def detect_from_segments(audio, override):
    if override != 'auto':
        return override
    current = detect_course(audio, override)
    if current != 'general':
        return current
    path = audio.with_suffix('.segments.json')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            data = data.get('segments', [])
        sample = ' '.join(str(s.get('text', '')) for s in data[:180]).lower()
    except Exception:
        return 'general'
    if sum(t in sample for t in ('calculator', 'compounding', 'present value', 'future value', 'pmt', 'interest rate', 'annuity')) >= 2:
        return 'math'
    if sum(t in sample for t in ('gantt', 'critical path', 'project scope', 'stakeholder')) >= 2:
        return 'project'
    if sum(t in sample for t in ('consumer behaviour', 'brand', 'marketing', 'segmentation')) >= 2:
        return 'marketing'
    if sum(t in sample for t in ('inventory', 'procurement', 'supply chain', 'logistics')) >= 2:
        return 'supply'
    return 'general'

def numeric_tokens(text):
    return re.findall(r'(?<![A-Za-z])\$?-?\d[\d,]*(?:\.\d+)?%?', text)

def transcript_anomaly(text):
    """High precision corruption detection; never silently edit source words."""
    t = str(text).strip()
    if not t:
        return 0, []
    reasons = []
    if '\ufffd' in t:
        reasons.append('replacement_character')
    # Multilingual-safe validation:
    # Chinese, Japanese and other Unicode scripts are valid speech text.
    # Script identity alone is not evidence of transcription corruption.
    # Keep independent corruption checks such as replacement characters.
    # Require multiple anomalous signals, not merely a name, a technical term, or a loanword.
    weird = re.findall(r'(?i)\b(?:execute disk disk|curriculum violating|ending agreement capital|foam)\b', t)
    if len(weird) >= 2:
        reasons.append('suspect_word_sequence')
    return (3 if 'replacement_character' in reasons else
            2 if reasons else 0), reasons


def index_tags(statement):
    """Narrow information index to meaningful educational navigation, not incidental numbers."""
    t = statement.lower()
    if transcript_anomaly(statement)[0] >= 2:
        return []
    assessment = bool(re.search(r'\b(?:quiz|test(?:s)?|exam(?:s)?|midterm|final grade|marks? off|half marks)\b', t))
    scheduling = bool(re.search(r'\b(?:no class|class(?:es)? (?:are )?(?:cancelled|canceled)|next week|in two weeks|due (?:on|by)|deadline|submit (?:by|before)|submission date)\b', t))
    exclusion = bool(re.search(r'\b(?:will not be (?:part|on|included|responsible)|not (?:be )?(?:part of|on (?:the )?(?:test|exam)|responsible for)|won.t be (?:on|part|responsible)|not going to (?:be )?responsible)\b', t))
    procedure = bool(re.search(r'\b(?:hit|press|enter|compute|calculate|set (?:p/y|c/y)|click|save)\b', t))
    numeric = bool(re.search(r'\$\s*\d|\b\d+(?:\.\d+)?\s*%', t))
    tags=[]
    if assessment: tags.append('ASSESSMENT')
    if scheduling: tags.append('SCHEDULE/DEADLINE')
    if exclusion: tags.append('EXCLUSION')
    if procedure and (assessment or scheduling or exclusion): tags.append('PROCEDURE')
    if numeric and (assessment or scheduling): tags.append('NUMERIC')
    return tags


def format_time(seconds):
    m, sec = divmod(int(seconds), 60)
    return f'{m:02d}:{sec:02d}'


def suspicious_segments(segments):
    return [(float(s.get('start', 0)), float(s.get('end', 0)),
             transcript_anomaly(s.get('text', ''))[1]) for s in segments
            if transcript_anomaly(s.get('text', ''))[0] >= 2]


def information_index(segments, max_items=32):
    """Conservative, timestamped navigation hints; never claim extracted facts are verified."""
    out = []
    for seg in segments:
        statement = ' '.join(str(seg.get('text', '')).split())
        if not statement:
            continue
        tags = index_tags(statement)
        if not tags:
            continue
        start = float(seg.get('start', 0))
        minutes, seconds = divmod(int(start), 60)
        excerpt = statement[:175] + ('...' if len(statement) > 175 else '')
        out.append((0 if any(t in tags for t in ('ASSESSMENT', 'SCHEDULE/DEADLINE', 'EXCLUSION')) else 1,
                    start, f'[{minutes:02d}:{seconds:02d}] [{"/".join(tags)}] {excerpt}'))
    out.sort(key=lambda row: (row[0], row[1]))
    chosen = sorted(out[:max_items], key=lambda row: row[1])
    return [row[2] for row in chosen], len(out)


def coverage_candidates(audio):
    """Read VAD-based candidate gaps from existing coverage QA; do not equate silence with missing speech."""
    path = audio.with_suffix('.coverage.json')
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        result = []
        for pair in data.get('potential_missing_regions', []):
            a, b = float(pair[0]), float(pair[1])
            if 0 <= a < b and b - a <= 30:
                result.append((a, b, 'VAD speech coverage candidate'))
        return result
    except (ValueError, TypeError, KeyError, IndexError):
        return []


def criticality(text):
    """Heuristic triage only; low scores never suppress existing QA findings."""
    score = 0
    if re.search(r'\b(?:test|exam|quiz|deadline|due|submit|cancelled|no class)\b', text, re.I):
        score += 4
    if re.search(r'\b(?:not|never|except|won.t|excluded|without)\b', text, re.I):
        score += 3
    if re.search(r'\$\s*\d|\b\d+(?:\.\d+)?\s*%', text):
        score += 2
    if re.search(r'\b(?:pmt|pv|fv|i/y|p/y|c/y|click|enter|save|compute)\b', text, re.I):
        score += 1
    return score


def select_targeted_regions(working_audio, report, max_regions):
    raw = json.loads(working_audio.with_suffix('.segments.json').read_text(encoding='utf-8'))
    segs = raw.get('segments', []) if isinstance(raw, dict) else raw
    items = []
    # VAD-detected uncovered speech is first priority. These are hypotheses, not confirmed omissions.
    for start, end, why in coverage_candidates(working_audio):
        items.append((-2, start, end, why))
    # Corrupt transcript windows are high-priority: verify, never silently overwrite.
    for start, end, reasons in suspicious_segments(segs):
        items.append((-1.5, start, end, 'transcript corruption: ' + ','.join(reasons)))
    # Existing calibration anomalies are high priority.
    for r in report.get('results', []):
        if not isinstance(r, dict):
            continue
        histories = [str(h.get('text', '')) for h in r.get('history', []) if isinstance(h, dict) and h.get('text') and not h.get('error')]
        numeric_conflict = len(histories) > 1 and any(numeric_tokens(x) != numeric_tokens(histories[-1]) for x in histories[:-1])
        if r.get('status') != 'LIKELY_ALREADY_TRANSCRIBED' or numeric_conflict:
            items.append((-1 if numeric_conflict else 0, float(r['start']), float(r['end']), 'calibration disagreement'))
    # Domain terminology anomalies: sample a bounded set, not every numeric occurrence.
    patterns = [
        (r'\b(?:catholic|pmp|\bFD\b|present valley|future valley|pavement)\b', 'possible technical term error'),
        (r'\b\d+(?:\.\d+)?\s*%|\$\s*\d', 'important numeric statement'),
    ]
    for seg in segs:
        t = str(seg.get('text', ''))
        for priority, (pattern, why) in enumerate(patterns, 2):
            if re.search(pattern, t, re.I):
                items.append((priority, float(seg['start']), float(seg['end']), why))
                break
    # Additional semantic risk checks across every course; no hard-coded course dependency.
    for seg in segs:
        statement = str(seg.get('text', ''))
        if criticality(statement) >= 4:
            items.append((1, float(seg['start']), float(seg['end']), 'critical-information risk'))
    items.sort(key=lambda x: (x[0], x[1]))
    selected = []
    for priority, start, end, why in items:
        if end <= start or end-start > 30:
            continue
        if any(start < other_end + 1 and end > other_start - 1 for other_start, other_end, _ in selected):
            continue
        selected.append((start, end, why))
        if len(selected) >= max_regions:
            break
    return selected

def targeted_verification(working_audio, report, course, limit):
    # Independent prompted re-ASR of suspicious windows; candidate only.
    if limit <= 0:
        return [], 0
    regions = select_targeted_regions(working_audio, report, limit)
    if not regions:
        return [], 0
    try:
        import numpy as np
        import mlx_whisper
    except Exception as exc:
        return [f'[Targeted re-ASR unavailable: {type(exc).__name__}: {exc}]'], 0
    raw = json.loads(working_audio.with_suffix('.segments.json').read_text(encoding='utf-8'))
    original_segments = raw.get('segments', []) if isinstance(raw, dict) else raw
    results = []
    completed = 0
    for start, end, reason in regions:
        left = max(0, start - 1.5)
        right = end + 1.5
        try:
            p = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(left), '-i', str(working_audio),
                                '-t', str(right-left), '-f', 'f32le', '-acodec', 'pcm_f32le',
                                '-ac', '1', '-ar', '16000', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            samples = np.frombuffer(p.stdout, dtype=np.float32).copy()
            if len(samples) < 1600:
                continue
            options = dict(path_or_hf_repo='mlx-community/whisper-large-v3-mlx', language=None,
                           task='transcribe', verbose=False, condition_on_previous_text=False,
                           temperature=0.0)
            if PROFILES[course]:
                options['initial_prompt'] = PROFILES[course]
            result = mlx_whisper.transcribe(samples, **options)
            candidate = ' '.join(str(s.get('text', '')).strip() for s in result.get('segments', []))
            candidate = candidate.strip()
            baseline = ' '.join(str(s.get('text', '')).strip() for s in original_segments
                                if float(s.get('start', 0)) < end and float(s.get('end', 0)) > start).strip()
            numeric_disagreement = bool(baseline and numeric_tokens(baseline) != numeric_tokens(candidate))
            comparison = (' [NUMERIC DISAGREEMENT — CHECK AUDIO]' if numeric_disagreement else '')
            if not candidate:
                results.append(f'[{start:.2f}s–{end:.2f}s] {reason}; re-ASR returned no speech')
            else:
                results.append(f'[{start:.2f}s–{end:.2f}s] {reason}{comparison}\n'
                               f'  Original (UNVERIFIED): {baseline[:350]}\n'
                               f'  Prompted re-ASR (UNVERIFIED): {candidate[:550]}')
            completed += 1
        except Exception as exc:
            results.append(f'[{start:.2f}s–{end:.2f}s] re-ASR failed ({type(exc).__name__}: {str(exc)[:150]}); check audio manually')
    return results, completed

def main():
    parser = argparse.ArgumentParser(description='Isolated transcription; publish one TXT only')
    parser.add_argument('audio', nargs='?')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument(
        '--no-organize',
        action='store_true',
        help='Keep final TXT beside source audio'
    )
    parser.add_argument('--qa-only', action='store_true', help='Not supported in zero-retention mode')
    parser.add_argument('--storage', action='store_true')
    parser.add_argument('--course', choices=['auto', *PROFILES.keys()], default='auto', help='Course terminology profile')
    parser.add_argument('--verify-regions', type=int, default=8, help='Maximum additional prompted audio checks (0 disables)')
    parser.add_argument('--no-index', action='store_true', help='Disable navigation index in TXT')
    parser.add_argument('--resume-job', type=str, help='Reuse retained failed job after verifying its audio and artifacts')
    args = parser.parse_args()
    if args.storage:
        for label, folder in [('Jobs', JOBS), ('Universal app data', ROOT)]:
            size = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) if folder.exists() else 0
            print(f'{label}: {size / 1024**2:.2f} MiB')
        return
    if not 0 <= args.verify_regions <= 30:
        parser.error('--verify-regions must be between 0 and 30')
    if args.qa_only:
        parser.error('--qa-only needs retained caches; V4 zero-retention mode requires a full run')
    if not args.audio:
        parser.error('Provide an audio path')
    original = Path(args.audio).expanduser().resolve()
    if not original.is_file():
        parser.error(f'Audio not found: {original}')
    if not ENGINE.is_file():
        parser.error(f'Engine missing: {ENGINE}')
    JOBS.mkdir(parents=True, exist_ok=True)
    if args.resume_job:
        job = Path(args.resume_job).expanduser().resolve()
        if job.parent != JOBS.resolve() or not job.is_dir() or not job.name.startswith('job-'):
            parser.error('--resume-job must be a retained LeoTranscribe Jobs/job-* directory')
        for ext in ('.segments.json', '.calibration.json', '.txt'):
            if not (job / original.name).with_suffix(ext).is_file():
                parser.error(f'Cannot resume: missing {ext} artifact')
        if not (job / original.name).is_file() or sha256(job / original.name) != sha256(original):
            parser.error('Cannot resume: retained audio does not match original')
    else:
        job = Path(tempfile.mkdtemp(prefix='job-', dir=JOBS))
    work_audio = job / original.name
    published = original.with_suffix('.txt')
    print('LeoTranscribe — timestamp integrity + resumable QA', flush=True)
    print('Temporary work is stored inside Application Support.', flush=True)
    try:
        if not args.resume_job:
            safe_copy(original, work_audio)
        if sha256(work_audio) != sha256(original):
            raise RuntimeError('Audio copy verification failed')
        course = detect_course(original, args.course)
        print(f'Course hint: {course}', flush=True)
        cmd = [sys.executable, str(ENGINE), str(work_audio)]
        if args.fresh:
            cmd.append('--fresh')
        env = os.environ.copy()
        if PROFILES[course]:
            env['LEOTRANSCRIBE_INITIAL_PROMPT'] = PROFILES[course]
        if args.resume_job:
            print('Resuming retained artifacts; skipping full Whisper transcription.', flush=True)
        else:
            subprocess.run(cmd, check=True, env=env)
        source_txt, report, count = verify(work_audio, original, False)
        body = source_txt.read_text(encoding='utf-8').rstrip() + '\n'
        raw_segments = json.loads(work_audio.with_suffix('.segments.json').read_text(encoding='utf-8'))
        segments = raw_segments.get('segments', []) if isinstance(raw_segments, dict) else raw_segments
        anomalies = suspicious_segments(segments)
        zero_duration = [(i, seg) for i, seg in enumerate(segments)
                         if float(seg['end']) == float(seg['start'])]
        if anomalies:
            print(f'Warning: {len(anomalies)} corrupt-looking transcript regions; see TXT QA flags.', flush=True)
        if not args.no_index:
            index_lines, total_candidates = information_index(segments)
            if index_lines:
                body = ('LECTURE INFORMATION INDEX — AUTOMATED NAVIGATION, NOT VERIFIED FACTS\n'
                        'Labels are heuristic; corrupted-looking lines excluded. Not audio-verified.\n'
                        + '\n'.join(index_lines) + '\n'
                        + f'[Showing {len(index_lines)} of {total_candidates} possible highlights]\n'
                        + '=' * 60 + '\nFULL TRANSCRIPT\n' + body)
        reviews, repeat_passes = extract_review(report)
        if zero_duration:
            reviews.insert(0, f'[TIMESTAMP WARNING: {len(zero_duration)} zero-duration segments; words retained; audio alignment NOT VERIFIED]')
            for i, seg in zero_duration:
                reviews.append(f'[segment #{i} at {float(seg["start"]):.2f}s] ZERO_DURATION_TIMESTAMP; audio verification needed; text retained: {str(seg.get("text", ""))[:250]}')
        if anomalies:
            reviews.insert(0, f'[TRANSCRIPT CORRUPTION: {len(anomalies)} suspicious regions; NOT silently replaced]')
            for a, b, why in anomalies:
                reviews.insert(1, f'[{a:.2f}s–{b:.2f}s] TRANSCRIPT_CORRUPTION; {", ".join(why)}; original text retained pending audio verification')

        course = detect_from_segments(work_audio, args.course)
        print(f'Verification profile: {course}; checking up to {args.verify_regions} audio regions...', flush=True)
        gap_count = len(coverage_candidates(work_audio))
        if gap_count:
            reviews.append(f'[VAD COVERAGE: {gap_count} potential uncovered speech intervals; candidate-only, not proof of missing words]')
        extra_reviews, checked = targeted_verification(work_audio, report, course, args.verify_regions)
        reviews.extend(extra_reviews)
        if reviews:
            body += ('\n' + '=' * 60 + '\n'
                     'WHISPER MULTI-PASS + TARGETED AUDIO VERIFICATION — UNRESOLVED FLAGS\n'
                     'Existing calibration repeat recognitions were compared before deletion.\n'
                     'Agreement is NOT proof of correctness. Candidates NEVER silently replace the transcript.\n'
                     + '\n'.join(reviews) + '\n')
        else:
            body += ('\n[Automated checks completed; this is not a guarantee '
                     'of word-for-word accuracy.]\n')
        # Atomic publication; preserve an existing TXT as a private backup.
        fd, temp = tempfile.mkstemp(prefix='.leotranscribe-', suffix='.tmp', dir=original.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            if published.exists():
                backups = ROOT / 'TranscriptBackups'
                backups.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime('%Y%m%d-%H%M%S')
                dest = backups / f'{original.stem}-{timestamp}-{os.getpid()}.txt'
                shutil.copy2(published, dest)
                if sha256(dest) != sha256(published):
                    raise RuntimeError('Previous TXT backup verification failed')
            os.replace(temp, published)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        if published.read_text(encoding='utf-8') != body:
            raise RuntimeError('Published TXT verification failed')
        print(f'\nVerified {count} segments structurally; review lines: {len(reviews)}; repeat Whisper passes audited: {repeat_passes}; targeted audio rechecks: {checked}')
        print(f'Final TXT: {published}')
        print('Deleting intermediate files for this completed job...', flush=True)
        shutil.rmtree(job)
        print('Done. No per-audio JSON or logs retained by this run.')

        # Smart Transcript Organizer (post-pipeline, non-critical)
        try:
            organizer = Path(__file__).resolve().parent / 'TranscriptOrganizer.py'
            if organizer.is_file() and not args.no_organize:
                subprocess.run(
                    [sys.executable, str(organizer), str(published)],
                    check=True,
                    timeout=300,
                )
            else:
                print('Organizer unavailable; TXT remains beside audio.')
        except Exception as exc:
            print(f'Organizer failed; original TXT preserved: {exc}')

    except Exception:
        print(f'\nFAILED: job retained for troubleshooting: {job}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
