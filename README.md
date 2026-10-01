# Clipmaker

Local tool: paste a video link (or upload a file) and get scored, captioned 9:16 clips plus an X thread, LinkedIn post and newsletter section.

## Setup (once)
1. Install Python 3.10+ and ffmpeg (`winget install ffmpeg` on Windows, `brew install ffmpeg` on Mac, `sudo apt install ffmpeg` on Linux).
2. `pip install -r requirements.txt`
3. Pick an AI:
   - Local: install Ollama, then `ollama pull llama3.1`.
   - Claude API (better clip picks): set the `ANTHROPIC_API_KEY` environment variable.

## Run on your computer
`python app.py`

The terminal prints two addresses:
- **This computer:** `http://localhost:5000`
- **Phone/tablet on the same Wi-Fi:** `http://YOUR-COMPUTER-LAN-IP:5000`

Keep the computer running while you use Clipmaker from your phone.

### YouTube login
If YouTube says you need to sign in or confirms you're not a bot:
1. Open YouTube in Chrome, Edge, Firefox, Brave, or another supported browser on the **same computer**.
2. Sign in to YouTube there.
3. In Clipmaker, choose that browser under **YouTube login**.
4. Start the job again.

You can also export a Netscape-format `cookies.txt` file and use the upload field as a fallback.

The first run downloads the Whisper speech model. Results are saved in the `jobs` folder.

## Phone access
The app listens on your local network by default. Your phone and computer must be connected to the same Wi-Fi.

If Windows Firewall asks whether Python may communicate on your network, allow it on your **Private network**. Do not expose port 5000 directly to the public internet.

You can change the bind host with `CLIPMAKER_HOST` and the port with `PORT`.
