"""Private single-user web UI for the Reframe content studio. No accounts or billing."""
import os, threading, uuid
from flask import Flask, render_template, request, jsonify, send_from_directory
import clipper_core

# Vercel's deployed filesystem is read-only; /tmp is the writable runtime area.
RUNTIME_DIR = os.environ.get("REFACTOR_RUNTIME_DIR", "/tmp/reframe")
UPLOAD_DIR = os.path.join(RUNTIME_DIR, "uploads")
CLIPS_DIR = os.path.join(RUNTIME_DIR, "clips")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

app = Flask(__name__)
# Keep this aligned with the intended local/private workflow. Platform request limits
# may still be lower than this value on serverless hosting.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

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
            min_len=60.0,
            max_len=90.0,
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
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    job_id = uuid.uuid4().hex[:12]
    mode = request.form.get("mode", "smart")
    captions = request.form.get("captions", "none")
    whisper_model = request.form.get("whisper_model", "small")
    raw_max = request.form.get("max_clips", "").strip()

    try:
        max_clips = int(raw_max) if raw_max else None
        if max_clips is not None and max_clips < 1:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Maximum clips must be a positive number."}), 400

    url = request.form.get("url", "").strip()
    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename:
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_upload_dir, exist_ok=True)
        input_path = os.path.join(
            job_upload_dir, os.path.basename(uploaded_file.filename)
        )
        uploaded_file.save(input_path)
        input_source = input_path
    elif url:
        input_source = url
    else:
        return jsonify({"error": "Add a video URL or upload a video file."}), 400

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "running",
            "log": [],
            "clips": [],
            "error": None,
        }

    params = {
        "mode": mode,
        "max_clips": max_clips,
        "captions": captions,
        "whisper_model": whisper_model,
    }
    threading.Thread(
        target=run_job, args=(job_id, input_source, params), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job"}), 404
        return jsonify(dict(job))


@app.route("/clips/<job_id>/<path:filename>")
def get_clip(job_id, filename):
    return send_from_directory(os.path.join(CLIPS_DIR, job_id), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Reframe on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
