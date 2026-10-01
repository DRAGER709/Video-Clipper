"""
clipper_core.py — lightweight video processing logic for the Reframe web app.
Uses yt-dlp and imageio-ffmpeg; no PyTorch, Whisper, OpenCV, NumPy, MoviePy,
or PySceneDetect dependencies are required.
"""

import os
import re
import shutil
import subprocess
import tempfile
import sys

import imageio_ffmpeg


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def ffmpeg_exe() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        path = shutil.which("ffmpeg")
        if path:
            return path
        raise RuntimeError("FFmpeg executable is unavailable.")


def check_ffmpeg():
    exe = ffmpeg_exe()
    result = subprocess.run([exe, "-version"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg could not be started.")
    return exe


def download_video(url: str, workdir: str, log=print) -> str:
    """Download a source video using the installed yt-dlp Python package."""
    out_path = os.path.join(workdir, "source.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        url,
    ]
    log(f"Downloading: {url}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"yt-dlp download failed{': ' + detail[-700:] if detail else '.'}")

    candidates = [
        os.path.join(workdir, f)
        for f in os.listdir(workdir)
        if f.startswith("source.")
    ]
    if not candidates:
        raise RuntimeError("Download finished but output file was not found.")
    mp4 = [f for f in candidates if f.lower().endswith(".mp4")]
    return mp4[0] if mp4 else candidates[0]


def get_duration(path: str) -> float:
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", path, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError("Could not determine video duration.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def cut_clip(src: str, start: float, end: float, out_path: str, reencode: bool = False):
    duration = end - start
    if reencode:
        cmd = [ffmpeg_exe(), "-y", "-ss", f"{start:.2f}", "-i", src,
               "-t", f"{duration:.2f}", "-c:v", "libx264", "-c:a", "aac",
               "-preset", "veryfast", out_path]
    else:
        cmd = [ffmpeg_exe(), "-y", "-ss", f"{start:.2f}", "-i", src,
               "-t", f"{duration:.2f}", "-c", "copy",
               "-avoid_negative_ts", "make_zero", out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def plan_fixed(total_duration: float, min_len: float, max_len: float):
    target = (min_len + max_len) / 2
    segments = []
    t = 0.0
    while t < total_duration - 1:
        end = min(t + target, total_duration)
        if end - t < min_len and segments:
            prev_start, _ = segments[-1]
            segments[-1] = (prev_start, total_duration)
            break
        segments.append((t, end))
        t = end
    return segments


def detect_scenes(src: str):
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", src,
           "-filter:v", "select='gt(scene,0.27)',showinfo",
           "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    timestamps = [0.0]
    for line in result.stderr.splitlines():
        marker = "pts_time:"
        if marker in line:
            try:
                value = float(line.split(marker, 1)[1].split()[0])
                if value > timestamps[-1] + 0.5:
                    timestamps.append(value)
            except (ValueError, IndexError):
                pass
    timestamps.append(get_duration(src))
    return list(zip(timestamps[:-1], timestamps[1:]))


def group_scenes_into_clips(scenes, min_len: float, max_len: float):
    if not scenes:
        return []
    clips = []
    cur_start, cur_end = scenes[0]
    for s, e in scenes[1:]:
        if e - cur_start <= max_len:
            cur_end = e
        else:
            if cur_end - cur_start >= min_len:
                clips.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_end - cur_start >= min_len:
        clips.append((cur_start, cur_end))
    return clips


def score_segment_loudness(src: str, start: float, end: float) -> float:
    cmd = [ffmpeg_exe(), "-i", src, "-ss", f"{start:.2f}",
           "-t", f"{end - start:.2f}", "-af", "volumedetect",
           "-vn", "-sn", "-dn", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    mean_db = -91.0
    for line in result.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.split("mean_volume:", 1)[1].replace("dB", "").strip())
            except ValueError:
                pass
    return mean_db


def transcribe_clip(clip_path: str, model_name: str, log=print):
    raise RuntimeError("Speech transcription is not included in the lightweight Vercel build.")


def run_pipeline(input_source, mode, output_dir, min_len=60.0, max_len=90.0,
                 max_clips=None, captions="none", whisper_model="small",
                 reencode=False, log=print):
    check_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        src = download_video(input_source, tmp, log=log) if is_url(input_source) else input_source
        if not os.path.exists(src):
            raise RuntimeError(f"Input file not found: {src}")

        duration = get_duration(src)
        log(f"Source duration: {duration/60:.1f} min")

        if mode == "fixed":
            segments = plan_fixed(duration, min_len, max_len)
        else:
            log("Detecting scenes (this scans the whole video, can take a while)...")
            scenes = detect_scenes(src)
            log(f"Found {len(scenes)} raw scenes")
            segments = group_scenes_into_clips(scenes, min_len, max_len)
            if mode == "smart":
                log("Scoring clips by audio energy...")
                scores = [score_segment_loudness(src, s, e) for s, e in segments]
                order = sorted(range(len(segments)), key=lambda i: scores[i], reverse=True)
                if max_clips:
                    order = order[:max_clips]
                segments = [segments[i] for i in sorted(order)]

        if mode != "smart" and max_clips:
            segments = segments[:max_clips]

        log(f"Cutting {len(segments)} clips...")
        for i, (start, end) in enumerate(segments, 1):
            filename = f"clip_{i:03d}_{int(start)}s-{int(end)}s.mp4"
            out_path = os.path.join(output_dir, filename)
            cut_clip(src, start, end, out_path, reencode=reencode)
            if captions != "none":
                log("Captions are unavailable in the lightweight Vercel build; continuing without captions.")
            log(f"Clip {i}/{len(segments)} done ({start:.0f}s-{end:.0f}s)")
            results.append({"filename": filename, "start": start, "end": end, "language": None})

    log("All clips done.")
    return results
