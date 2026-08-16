#!/usr/bin/env python3
"""
video_clipper.py — Turn a long video (local file or YouTube URL) into a
batch of 60-90 second clips.

Modes:
  fixed  - dumb, fast: cut every N seconds (default ~75s), no analysis.
  scene  - detects hard cuts/scene changes (PySceneDetect) and groups
           them into clips as close to 60-90s as possible, so cuts land
           on actual scene boundaries instead of mid-shot.
  smart  - like scene, but also scores each scene by audio loudness/
           energy (a cheap proxy for "something is happening": action,
           raised voices, music swells) and picks the highest-scoring
           windows first. Good for auto-generating "highlights".

Captions:
    Auto-captions are burned directly into each clip using OpenAI's Whisper
    (runs locally, no API key, no internet needed once the model is
    downloaded). Whisper auto-detects the spoken language per clip and
    captions in that same language — it supports ~99 languages, so this
    works regardless of what language the source video is in.

Requirements (install once):
    pip install yt-dlp scenedetect[opencv] moviepy numpy openai-whisper
    # ffmpeg must be installed and on PATH (https://ffmpeg.org/download.html)

Examples:
    # Local file, dumb fixed-length split, no captions
    python video_clipper.py --input movie.mp4 --mode fixed --captions none

    # YouTube URL, cut on scene boundaries, burn in auto-captions (default)
    python video_clipper.py --input "https://youtube.com/watch?v=XXXX" --mode scene

    # YouTube URL, "smart" highlight-style picks, only keep top 8 clips
    python video_clipper.py --input "https://youtube.com/watch?v=XXXX" --mode smart --max-clips 8
"""

import argparse
import math
import os
import re
import subprocess
import sys
import shutil
import tempfile


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def check_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ERROR: ffmpeg/ffprobe not found on PATH. Install ffmpeg first.")


def download_video(url: str, workdir: str) -> str:
    """Download a YouTube (or other yt-dlp supported) URL to a local mp4."""
    if not shutil.which("yt-dlp"):
        sys.exit("ERROR: yt-dlp not installed. Run: pip install yt-dlp")
    out_path = os.path.join(workdir, "source.mp4")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        url,
    ]
    print(f"[download] {url}")
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_path):
        # yt-dlp sometimes names it differently if merge wasn't needed
        candidates = [f for f in os.listdir(workdir) if f.startswith("source")]
        if candidates:
            out_path = os.path.join(workdir, candidates[0])
        else:
            sys.exit("ERROR: download finished but output file not found.")
    return out_path


def get_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def cut_clip(src: str, start: float, end: float, out_path: str, reencode: bool = False):
    """Cut [start, end) from src into out_path.
    Stream-copy is fast but can be a bit imprecise at cut points (snaps to
    nearest keyframe). Re-encode is frame-accurate but slower.
    """
    duration = end - start
    if reencode:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", src,
            "-t", f"{duration:.2f}",
            "-c:v", "libx264", "-c:a", "aac", "-preset", "veryfast",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", src,
            "-t", f"{duration:.2f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            out_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True)


# --------------------------------------------------------------------------
# Mode: fixed
# --------------------------------------------------------------------------

def plan_fixed(total_duration: float, min_len: float, max_len: float):
    """Just walk forward in chunks near the middle of [min_len, max_len]."""
    target = (min_len + max_len) / 2
    segments = []
    t = 0.0
    while t < total_duration - 1:
        end = min(t + target, total_duration)
        if end - t < min_len and segments:
            # too short a tail clip -> merge into previous instead of keeping a stub
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, total_duration)
            break
        segments.append((t, end))
        t = end
    return segments


# --------------------------------------------------------------------------
# Mode: scene / smart (shared scene-detection step)
# --------------------------------------------------------------------------

def detect_scenes(src: str):
    """Returns list of (start, end) scene boundaries using PySceneDetect."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        sys.exit("ERROR: scene/smart mode needs PySceneDetect. Run: pip install scenedetect[opencv]")

    video = open_video(src)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def group_scenes_into_clips(scenes, min_len: float, max_len: float):
    """Greedily merge consecutive scenes until adding the next one would
    overshoot max_len; then cut. Ensures every clip lands on a real scene
    boundary rather than mid-shot."""
    if not scenes:
        return []
    clips = []
    cur_start = scenes[0][0]
    cur_end = scenes[0][1]
    for (s, e) in scenes[1:]:
        if (e - cur_start) <= max_len:
            cur_end = e
        else:
            if (cur_end - cur_start) < min_len:
                # short clip: keep it anyway rather than dropping content
                pass
            clips.append((cur_start, cur_end))
            cur_start = s
            cur_end = e
    clips.append((cur_start, cur_end))
    return clips


def score_segment_loudness(src: str, start: float, end: float) -> float:
    """Cheap 'interestingness' proxy: mean audio volume (RMS-ish) over the
    segment, via ffmpeg's volumedetect filter. Louder/more dynamic audio
    (action, raised voices, music) scores higher. Not true content
    understanding — just a fast, dependency-light heuristic."""
    cmd = [
        "ffmpeg", "-i", src, "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
        "-af", "volumedetect", "-vn", "-sn", "-dn", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    mean_db = -91.0  # silence fallback
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.strip().split("mean_volume:")[1].replace("dB", "").strip())
            except ValueError:
                pass
    return mean_db  # closer to 0 = louder = higher score


# --------------------------------------------------------------------------
# Captions (Whisper: local, auto language detection)
# --------------------------------------------------------------------------

_WHISPER_MODEL_CACHE = {}


def load_whisper_model(model_name: str):
    """Load (and cache) a Whisper model so it's only loaded once per run."""
    if model_name in _WHISPER_MODEL_CACHE:
        return _WHISPER_MODEL_CACHE[model_name]
    try:
        import whisper
    except ImportError:
        sys.exit("ERROR: captions need openai-whisper. Run: pip install openai-whisper")
    print(f"[captions] loading whisper model '{model_name}' (first run downloads it, ~1-2GB for larger sizes)...")
    model = whisper.load_model(model_name)
    _WHISPER_MODEL_CACHE[model_name] = model
    return model


def transcribe_clip(clip_path: str, model_name: str):
    """Auto-detects language and transcribes IN that language (no
    translation). Returns (segments, detected_language_code)."""
    model = load_whisper_model(model_name)
    # task='transcribe' (not 'translate') keeps captions in the original
    # spoken language; language=None lets Whisper auto-detect it.
    result = model.transcribe(clip_path, task="transcribe", language=None, verbose=False)
    return result.get("segments", []), result.get("language", "unknown")


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, srt_path: str):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def _escape_path_for_ffmpeg_filter(path: str) -> str:
    """ffmpeg's -vf subtitles=... filter treats ':' and '\\' specially,
    which breaks on Windows paths (C:\\...). Escape it as the filter
    expects."""
    p = os.path.abspath(path)
    p = p.replace("\\", "/")
    p = p.replace(":", "\\:")
    return p


def burn_captions(video_path: str, srt_path: str, out_path: str):
    """Re-encodes video with the SRT hard-burned into the frame."""
    escaped = _escape_path_for_ffmpeg_filter(srt_path)
    style = "FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-c:v", "libx264", "-c:a", "copy", "-preset", "veryfast",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def add_captions_to_clip(clip_path: str, model_name: str, mode: str):
    """mode: 'burn' (captions baked into video, replaces clip_path) or
    'srt' (video untouched, .srt written alongside it).
    Returns the detected language code."""
    segments, language = transcribe_clip(clip_path, model_name)
    srt_path = os.path.splitext(clip_path)[0] + ".srt"
    write_srt(segments, srt_path)

    if mode == "burn":
        tmp_out = clip_path + ".captioned.mp4"
        burn_captions(clip_path, srt_path, tmp_out)
        os.replace(tmp_out, clip_path)
        os.remove(srt_path)  # captions are now baked in, no need for the sidecar file
    # mode == 'srt' -> leave the .srt file next to the clip, video untouched

    return language


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Split a long video into 60-90s clips.")
    ap.add_argument("--input", required=True, help="Local video path or YouTube/URL")
    ap.add_argument("--mode", choices=["fixed", "scene", "smart"], default="fixed")
    ap.add_argument("--min-len", type=float, default=60.0)
    ap.add_argument("--max-len", type=float, default=90.0)
    ap.add_argument("--max-clips", type=int, default=None,
                     help="Only keep this many clips (highest-scoring first in smart mode)")
    ap.add_argument("--output-dir", default="./clips")
    ap.add_argument("--reencode", action="store_true",
                     help="Frame-accurate cuts (slower). Default is fast stream-copy.")
    ap.add_argument("--captions", choices=["burn", "srt", "none"], default="burn",
                     help="'burn' hardcodes captions into the video (default), "
                          "'srt' writes a separate .srt per clip, 'none' skips captions.")
    ap.add_argument("--whisper-model", default="small",
                     choices=["tiny", "base", "small", "medium", "large"],
                     help="Bigger = more accurate but slower. 'small' is a good default.")
    args = ap.parse_args()

    check_ffmpeg()
    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        src = download_video(args.input, tmp) if is_url(args.input) else args.input
        if not os.path.exists(src):
            sys.exit(f"ERROR: input file not found: {src}")

        duration = get_duration(src)
        print(f"[info] source duration: {duration/60:.1f} min")

        if args.mode == "fixed":
            segments = plan_fixed(duration, args.min_len, args.max_len)
            scores = [0.0] * len(segments)

        else:  # scene or smart
            print("[info] detecting scenes...")
            scenes = detect_scenes(src)
            print(f"[info] found {len(scenes)} raw scenes")
            segments = group_scenes_into_clips(scenes, args.min_len, args.max_len)

            if args.mode == "smart":
                print("[info] scoring clips by audio energy...")
                scores = [score_segment_loudness(src, s, e) for (s, e) in segments]
                # sort segments by score (loudest first) but remember original order for naming
                order = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)
                if args.max_clips:
                    order = order[: args.max_clips]
                segments = [segments[i] for i in sorted(order)]  # keep chronological order in output
            else:
                scores = [0.0] * len(segments)

        if args.mode != "smart" and args.max_clips:
            segments = segments[: args.max_clips]

        print(f"[info] cutting {len(segments)} clips into {args.output_dir}/")
        for i, (start, end) in enumerate(segments, 1):
            out_path = os.path.join(args.output_dir, f"clip_{i:03d}_{int(start)}s-{int(end)}s.mp4")
            cut_clip(src, start, end, out_path, reencode=args.reencode)
            msg = f"  clip {i}: {start:.1f}s -> {end:.1f}s  ({end-start:.1f}s)  saved to {out_path}"

            if args.captions != "none":
                lang = add_captions_to_clip(out_path, args.whisper_model, args.captions)
                msg += f"  [captions: {lang}]"

            print(msg)

    print("[done]")


if __name__ == "__main__":
    main()
