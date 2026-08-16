"""
app.py — local web UI for video_clipper.

Run with:  python app.py
Then open: http://localhost:5000 in your browser.
"""

import os
import threading
import time
import uuid
from functools import wraps

from flask import Flask, render_template, request, jsonify, send_from_directory, Response

import clipper_core

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# --- password protection ---
# Set these as environment variables on your host (Render dashboard ->
# Environment). Never commit real credentials into a public git repo —
# these fallback values are only for local testing.
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "changeme123")


def check_auth(username, password):
    return username == AUTH_USERNAME and password == AUTH_PASSWORD


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                "Login required.", 401,
                {"WWW-Authenticate": 'Basic realm="Clip Bench"'},
            )
        return f(*args, **kwargs)
    return decorated


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8GB upload cap

# In-memory job store. Fine for a single-user local tool; not meant to
# survive a restart or serve multiple people at once.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _update_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _append_log(job_id, message):
    with JOBS_LOCK:
        JOBS[job_id]["log"].append(message)


def run_job(job_id, input_source, params):
    output_dir = os.path.join(CLIPS_DIR, job_id)
    try:
        results = clipper_core.run_pipeline(
            input_source=input_source,
            mode=params["mode"],
            output_dir=output_dir,
            min_len=params["min_len"],
            max_len=params["max_len"],
            max_clips=params["max_clips"],
            captions=params["captions"],
            whisper_model=params["whisper_model"],
            reencode=False,
            log=lambda msg: _append_log(job_id, msg),
        )
        _update_job(job_id, status="done", clips=results)
    except Exception as e:
        _append_log(job_id, f"ERROR: {e}")
        _update_job(job_id, status="error", error=str(e))


@app.route("/")
@requires_auth
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
@requires_auth
def process():
    job_id = uuid.uuid4().hex[:12]

    mode = request.form.get("mode", "fixed")
    captions = request.form.get("captions", "burn")
    whisper_model = request.form.get("whisper_model", "tiny")
    max_clips = request.form.get("max_clips", "").strip()
    max_clips = int(max_clips) if max_clips else None

    url = request.form.get("url", "").strip()
    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename:
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_upload_dir, exist_ok=True)
        input_path = os.path.join(job_upload_dir, uploaded_file.filename)
        uploaded_file.save(input_path)
        input_source = input_path
    elif url:
        input_source = url
    else:
        return jsonify({"error": "Provide either a file or a URL."}), 400

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "clips": [], "error": None}

    params = {
        "mode": mode,
        "min_len": 60.0,
        "max_len": 90.0,
        "max_clips": max_clips,
        "captions": captions,
        "whisper_model": whisper_model,
    }
    thread = threading.Thread(target=run_job, args=(job_id, input_source, params), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
@requires_auth
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(dict(job))


@app.route("/clips/<job_id>/<path:filename>")
@requires_auth
def get_clip(job_id, filename):
    return send_from_directory(os.path.join(CLIPS_DIR, job_id), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Video Clipper on port {port}")
    print(f"Login: username='{AUTH_USERNAME}'  password='{AUTH_PASSWORD}'")
    app.run(host="0.0.0.0", port=port, debug=False)
