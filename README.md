# Clipmaker

Local tool: paste a video link (or upload a file) and get scored, captioned 9:16 clips plus an X thread, LinkedIn post and newsletter section.

## Setup (once)
1. Install Python 3.10+ and ffmpeg (`winget install ffmpeg` on Windows, `brew install ffmpeg` on Mac, `sudo apt install ffmpeg` on Linux).
2. `pip install -r requirements.txt`
3. Pick an AI:
   - Local: install Ollama, then `ollama pull llama3.1`.
   - Claude API (better clip picks): set the `ANTHROPIC_API_KEY` environment variable.

## Run
`python app.py`, then open http://localhost:5000

Results are saved in the `jobs` folder. The first run downloads the Whisper speech model.