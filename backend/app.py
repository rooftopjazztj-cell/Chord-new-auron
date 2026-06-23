"""Lux Chord Studio — transcription & analysis backend.

Endpoints:
  GET  /                  Friendly root
  GET  /health            Cheap liveness check (no models loaded)
  GET  /info              Capability report (which engines are available)
  GET  /build-info        Diagnostic: confirms which build Railway is running
  POST /analyze/full      Full pipeline: ingest → (separate) → structure → chords → align → (validate)
  POST /analyze/structure Only beats / downbeats / key
  POST /analyze/chords    Only chord events
  POST /youtube/fetch     Download YouTube → mp3 (internal)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import librosa
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services import ingest
from services.align import Bar, align_to_bars
from services.chords import detect_chords
from services.spotify import lookup as spotify_lookup
from services.structure import detect_beats_and_downbeats, detect_key, detect_tempo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("lux")

# GitHub sync marker: Lux Chord Studio API 2.0.0 — 2026-06-11
app = FastAPI(title="Lux Chord Studio API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    logger.info("[%s] -> %s %s", rid, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception as e:  # noqa: BLE001
        logger.exception("[%s] !! %s", rid, e)
        return JSONResponse({"error": str(e), "request_id": rid}, status_code=500)
    dt = (time.time() - t0) * 1000
    logger.info("[%s] <- %s %s (%.0f ms)", rid, response.status_code, request.url.path, dt)
    response.headers["x-request-id"] = rid
    return response


# ────────────────────────────────────────────────────────────────────────────
# Meta endpoints
# ────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"service": "lux-chord-studio", "version": "2.0.0", "ok": True}


@app.get("/health")
def health():
    return {"ok": True, "service": "lux-chord-studio", "version": "2.0.0"}


@app.get("/build-info")
def build_info():
    """Temporary diagnostic: confirms exactly which build Railway is running."""
    return {
        "service": "lux-chord-studio",
        "version": "2.0.0",
        "git_commit": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "unknown"),
        "git_branch": os.environ.get("RAILWAY_GIT_BRANCH", "unknown"),
        "git_repo": os.environ.get("RAILWAY_GIT_REPO_NAME", "unknown"),
        "build_time": os.environ.get("RAILWAY_DEPLOYMENT_ID", "unknown"),
        "endpoints": [r.path for r in app.routes if hasattr(r, "path")],
    }


@app.get("/info")
def info():
    caps = {
        "madmom": _try_import("madmom"),
        "demucs": _try_import("demucs"),
        "vamp": _try_import("vamp"),
        "yt_dlp": _try_import("yt_dlp"),
        "spotipy": _try_import("spotipy"),
    }

    return {
        "ok": True,
        "capabilities": caps,
        "spotify_configured": bool(os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")),
    }


def _try_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


# ────────────────────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────────────────────


class YouTubeRequest(BaseModel):
    youtube_url: str


class AnalyzeFullResponse(BaseModel):
    key: Optional[str]
    bpm: Optional[float]
    timeSignature: int
    beats: List[float]
    downbeats: List[float]
    bars: List[Bar]
    chords: list
    duration: float
    source: str
    spotify: Optional[dict] = None


# ────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ────────────────────────────────────────────────────────────────────────────


def _resolve_audio(file: Optional[UploadFile], youtube_url: Optional[str]) -> Path:
    if file is not None:
        data = file.file.read()
        suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
        return ingest.save_upload(data, suffix=suffix)
    if youtube_url:
        return ingest.download_youtube(youtube_url)
    raise HTTPException(400, "Provide a file upload or a youtube_url")


def _maybe_separate(audio_path: Path, do_separate: bool) -> Path:
    """If separation is enabled, write the harmonic mix to a new WAV and
    return that path. Otherwise return the original."""
    if not do_separate:
        return audio_path
    try:
        import soundfile as sf  # type: ignore

        from services.separation import separate_harmonic

        mono, sr = separate_harmonic(audio_path)
        out = audio_path.with_suffix(".harmonic.wav")
        sf.write(str(out), mono, sr)
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("Source separation skipped (%s); using full mix.", e)
        return audio_path


@app.post("/analyze/full")
def analyze_full(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    separate: bool = Query(True, description="Run Demucs harmonic separation first"),
    validate_with_spotify: bool = Query(True),
    spotify_query: Optional[str] = Query(None, description="Track title / artist, or Spotify URL"),
    max_chords_per_bar: int = Query(3, ge=1, le=4),
):
    audio = _resolve_audio(file, youtube_url)
    try:
        analysis_path = _maybe_separate(audio, do_separate=separate)
        duration = float(librosa.get_duration(path=str(analysis_path)))

        # Structure (use the original mix for beats — percussion helps).
        beats, downbeats, bpb = detect_beats_and_downbeats(audio)
        key = detect_key(audio)
        tempo = detect_tempo(audio)

        # Chords (use the harmonic-only mix when available).
        chord_events = detect_chords(analysis_path)

        # Align chords to bars.
        bars = align_to_bars(
            chord_events,
            beats=beats,
            downbeats=downbeats,
            beats_per_bar=bpb,
            audio_duration=duration,
            max_chords_per_bar=max_chords_per_bar,
        )

        # Optional Spotify validation.
        spotify = None
        if validate_with_spotify and spotify_query:
            spotify = spotify_lookup(spotify_query)
            if spotify:
                if spotify.get("tempo"):
                    tempo = float(spotify["tempo"])
                if spotify.get("key"):
                    key = f"{spotify['key']} {spotify.get('mode', 'major')}"

        return {
            "key": key,
            "bpm": tempo,
            "timeSignature": bpb,
            "beats": beats,
            "downbeats": downbeats,
            "bars": bars,
            "chords": chord_events,
            "duration": duration,
            "source": "spotify-validated" if spotify else "audio",
            "spotify": spotify,
        }
    finally:
        ingest.cleanup(audio)


@app.post("/analyze/structure")
def analyze_structure(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
):
    audio = _resolve_audio(file, youtube_url)
    try:
        beats, downbeats, bpb = detect_beats_and_downbeats(audio)
        return {
            "key": detect_key(audio),
            "bpm": detect_tempo(audio),
            "timeSignature": bpb,
            "beats": beats,
            "downbeats": downbeats,
        }
    finally:
        ingest.cleanup(audio)


@app.post("/analyze/chords")
def analyze_chords(
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    separate: bool = Query(False),
):
    audio = _resolve_audio(file, youtube_url)
    try:
        analysis_path = _maybe_separate(audio, do_separate=separate)
        return {"chords": detect_chords(analysis_path)}
    finally:
        ingest.cleanup(audio)


@app.post("/youtube/fetch")
def youtube_fetch(req: YouTubeRequest):
    try:
        path = ingest.download_youtube(req.youtube_url)
        return {"ok": True, "path": str(path)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"yt-dlp failed: {e}")
