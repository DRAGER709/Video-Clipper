# Deploy target: Render (or any host that runs a Dockerfile).
# ffmpeg isn't available on most Python hosting platforms by default,
# so we install it explicitly at the OS level here.

FROM python:3.11-slim

# ffmpeg = video cutting/audio. git = needed by some yt-dlp/whisper installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the 'tiny' Whisper model into the image so the first user
# request isn't stuck waiting on a ~75MB download + load. If you upgrade
# the free tier's RAM, you can also pre-cache 'small' here the same way.
RUN python -c "import whisper; whisper.load_model('tiny')"

COPY . .

# Render sets $PORT itself; app.py already reads it via os.environ.
CMD ["python", "app.py"]
