# Video Clipper — setup & usage

## 1. Install requirements (one time)
```bash
pip install yt-dlp scenedetect[opencv] moviepy numpy openai-whisper
```
Also make sure **ffmpeg** is installed and on your PATH:
- Windows: https://ffmpeg.org/download.html (or `choco install ffmpeg`)
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## 2. Run it

**Dumb, fast, fixed-length split** (every ~75s, no analysis):
```bash
python video_clipper.py --input movie.mp4 --mode fixed
```

**Cut on real scene boundaries** (no mid-shot cuts):
```bash
python video_clipper.py --input "https://youtube.com/watch?v=XXXX" --mode scene
```

**"Smart" mode** — scores each scene by audio loudness (a fast proxy for
action/dialogue/music — not true AI content understanding) and keeps the
most "eventful" clips:
```bash
python video_clipper.py --input "https://youtube.com/watch?v=XXXX" --mode smart --max-clips 8
```

Clips land in `./clips/` by default (`--output-dir` to change it), named
like `clip_001_0s-75s.mp4`.

## Captions
By default (`--captions burn`) every clip gets auto-generated captions
burned directly into the video, using OpenAI's Whisper running locally —
no API key, no internet needed after the model's first download.

- **Language**: Whisper auto-detects the spoken language per clip and
  captions it in that *same* language (no translation). It supports
  ~99 languages, so this works whatever language the source is in. Mixed-
  language clips get captioned in whichever language Whisper detects as
  dominant for that clip.
- `--captions srt` — skip burning in, just write a `.srt` file next to
  each clip (editable, good if you want to style captions yourself later).
- `--captions none` — no captions at all, fastest.
- `--whisper-model` — `tiny`/`base`/`small`/`medium`/`large`. Default is
  `small` (good accuracy/speed balance). Bigger models are slower but more
  accurate, especially for less common languages or noisy audio.
- The **first run downloads the model** (small ≈ 460MB, large ≈ 1.5GB) —
  needs internet once, then it's cached locally.
- Captioning adds real time per clip (transcription + a second ffmpeg
  encode pass), so a 120-min movie split into ~15 clips with captions will
  take noticeably longer than `--captions none`.

## Notes / honest limitations
- **`fixed`** is pure math — instant, but cuts can land mid-sentence or mid-scene.
- **`scene`** is much better for clip quality (real cut points) but needs
  `scenedetect` and takes longer on a 120-min movie (it scans the whole video).
- **`smart`** is scene detection + a loudness heuristic, *not* an AI that
  understands plot, dialogue, or "best moments." It's a decent proxy for
  finding punchy/high-energy segments, but it'll sometimes pick a loud
  action beat over a quiet-but-important dialogue scene. If you want true
  content-aware highlight picking (e.g. "find the funniest moment" or
  "find the twist"), that needs a transcript + an LLM pass on top of this
  — happy to add that as a `--mode ai` option if you want it, it just
  needs an API key for transcription (e.g. Whisper) and an LLM call.
- Default cuts use fast stream-copy (`-c copy`), which snaps to the
  nearest keyframe — clip boundaries can be off by up to ~1-2s. Pass
  `--reencode` for frame-accurate cuts (slower, re-encodes video).
- A 120-minute movie in `scene`/`smart` mode can take several minutes to
  process (scene detection scans every frame). `fixed` mode is near-instant.
