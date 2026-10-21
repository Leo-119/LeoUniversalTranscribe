# LeoUniversalTranscribe

An Apple Silicon-accelerated transcription pipeline for local audio, video, and YouTube content.

## Overview

LeoUniversalTranscribe combines multiple speech-processing components into a unified transcription workflow.

The system is designed to support long-form recordings, automated transcription, speaker diarization, and additional quality-control passes.

## Features

- Unified command-line entry point
- Local audio and video processing
- YouTube audio extraction
- MLX Whisper Large-v3 speech recognition
- Silero VAD speech segmentation
- Pyannote speaker diarization
- Multi-pass transcription verification
- Timestamp integrity checks
- Resumable processing
- Organized TXT transcript output

## Architecture

The pipeline consists of:

1. Input detection and preprocessing
2. Voice activity detection
3. Speech recognition
4. Speaker diarization
5. Internal quality assurance
6. Coverage and calibration checks
7. Transcript publication and organization

## Requirements

- macOS on Apple Silicon
- Python 3.13-compatible environment
- FFmpeg
- yt-dlp
- Python dependencies including mlx-whisper, torch, silero-vad, and pyannote-audio

## Project Structure

The `src/` directory contains the transcription pipeline and supporting modules.

## Status

Development release. Installation instructions and cross-machine testing are in progress.

## Privacy

Audio recordings and transcripts are processed locally, although model downloads and YouTube retrieval require network access.

Users should review the privacy policies of any external services they access.

## License

License to be determined.
