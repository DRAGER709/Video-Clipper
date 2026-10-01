#!/usr/bin/env python3
"""Clipmaker: video link or file -> scored, captioned 9:16 clips + posts in your voice.
Run:  python app.py   then open http://localhost:5000"""
import json, os, re, subprocess, threading, uuid
from pathlib import Path
import requests
from flask import Flask, request, jsonify, send_from_directory, Response

ROOT = Path(os.environ.get("CLIPMAKER_DATA_DIR", "/tmp/jobs" if os.environ.get("VERCEL") else str(Path(__file__).parent / "jobs")))
ROOT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
JOBS = {}

# ---------- LLM ----------
def llm(cfg, prompt, system="You are a precise assistant. Reply with valid JSON only."):
    if cfg["provider"] == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=300,
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            json={"model": cfg["model"], "max_tokens": 4000, "system": system,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    r = requests.post("http://localhost:11434/api/chat", timeout=900,
        json={"model": cfg["model"], "stream": False, "format": "json",
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"]

def parse_json(t):
    return json.loads(re.search(r"[\[{].*[\]}]", t, re.S).group(0))

# ---------- pipeline steps ----------
def download(url, d, cookies_file=None, user_agent=""):
    import yt_dlp
    opts = {
        "outtmpl": str(d / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
    }
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)
    if user_agent.strip():
        opts["http_headers"] = {"User-Agent": user_agent.strip()}
    if "youtube.com" in url or "youtu.be" in url:
        opts["extractor_args"] = {"youtube": {"player_client": ["web_embedded", "mweb", "tv"]}}
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    files = [p for p in d.glob("source.*") if p.suffix.lower() not in {".part", ".ytdl"}]
    if not files:
        raise RuntimeError("Download finished without producing a video file")
    return files[0]

def transcribe(path, size):
    from faster_whisper import WhisperModel
    model = WhisperModel(size, compute_type="int8")
    segs, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
    return [{"start": s.start, "end": s.end, "text": s.text.strip(),
             "words": [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in (s.words or [])]} for s in segs]

def pick_clips(cfg, segs, n):
    chunks, cur, size = [], [], 0
    for s in segs:
        line = f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
        cur.append(line); size += len(line)
        if size > 9000:
            chunks.append("\n".join(cur)); cur, size = [], 0
    if cur:
        chunks.append("\n".join(cur))
    cands = []
    for c in chunks:
        p = ('Find up to 4 moments in this transcript that work as standalone short-form clips '
             '(25-60 seconds, strong hook in the first sentence, one complete thought). Timestamps are seconds. '
             'Score each 0-100 for viral potential. Return JSON: '
             '{"clips":[{"start":0.0,"end":0.0,"title":"short headline","score":0,"reason":"one sentence"}]}\n\n' + c)
        try:
            r = parse_json(llm(cfg, p))
            cands += r["clips"] if isinstance(r, dict) else r
        except Exception:
            continue
    starts, ends = [s["start"] for s in segs], [s["end"] for s in segs]
    good = []
    for c in cands:
        try:
            a = min(starts, key=lambda x: abs(x - float(c["start"])))
            b = min(ends, key=lambda x: abs(x - float(c["end"])))
            if 10 <= b - a <= 90:
                good.append({**c, "start": a, "end": b, "score": int(c.get("score", 50))})
        except Exception:
            pass
    good.sort(key=lambda c: -c["score"])
    out = []
    for c in good:
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out

def ts(t):
    t = max(0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"

def make_ass(words, t0, path):
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
            "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            "Style: D,Arial,80,&H0000E5FF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines = []
    for i in range(0, len(words), 4):
        g = words[i:i + 4]
        parts = []
        for j, w in enumerate(g):
            end = g[j + 1]["s"] if j + 1 < len(g) else w["e"]
            word = re.sub(r"[{}\\]", "", w["w"]).upper()
            parts.append("{\\k%d}%s" % (max(1, round((end - w["s"]) * 100)), word))
        lines.append(f"Dialogue: 0,{ts(g[0]['s'] - t0)},{ts(g[-1]['e'] - t0)},D,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text(head + "\n".join(lines), encoding="utf-8")

def render(src, c, words, d, i, layout):
    make_ass(words, c["start"], d / f"clip{i}.ass")
    ass = f"ass=clip{i}.ass"
    if layout == "fit":
        fc = ("[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30[bg];"
              f"[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,{ass}[v]")
    else:
        fc = f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,{ass}[v]"
    cmd = ["ffmpeg", "-y", "-ss", f"{c['start']:.2f}", "-t", f"{c['end'] - c['start']:.2f}", "-i", str(Path(src).resolve()),
           "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", f"clip{i}.mp4"]
    try:
        subprocess.run(cmd, cwd=d, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg failed: " + e.stderr.decode(errors="ignore")[-400:])

def write_posts(cfg, segs, voice):
    text = " ".join(s["text"] for s in segs)
    if len(text) > 30000:
        text = text[:15000] + " ... " + text[-15000:]
    v = (f"Match the voice of these writing samples (tone, sentence length, vocabulary):\n{voice}\n\n"
         if voice.strip() else "Write in a clear, direct, conversational voice.\n\n")
    p = (v + "From this transcript write: an X thread (5-8 posts, each under 270 characters, first one is a hook), "
         "one LinkedIn post (120-220 words, short paragraphs), and a newsletter section (150-250 words with a heading). "
         'Use only ideas from the transcript. Return JSON: {"x_thread":["..."],"linkedin":"...","newsletter":"..."}'
         "\n\nTranscript:\n" + text)
    return parse_json(llm(cfg, p, "You write social posts. Reply with valid JSON only."))

# ---------- job runner ----------
def run(jid, f, src):
    j, d = JOBS[jid], ROOT / jid
    def log(m, pct):
        j["log"].append(m); j["pct"] = pct
    try:
        cfg = {"provider": f.get("provider", "ollama"), "model": f.get("model") or "llama3.1"}
        if src is None:
            log("Downloading video", 5)
            cookies_file = None
            if f.get("cookies_upload"):
                cookies_file = d / "cookies.txt"
                cookies_file.write_text(f["cookies_upload"], encoding="utf-8")
            src = download(f["url"], d, cookies_file, f.get("user_agent", ""))
        log("Transcribing (first run downloads the Whisper model)", 20)
        segs = transcribe(src, f.get("whisper", "small"))
        if not segs:
            raise RuntimeError("No speech found in this video")
        log("Finding the best moments", 45)
        picks = pick_clips(cfg, segs, int(f.get("n", 5)))
        if not picks:
            raise RuntimeError("The model returned no usable clips. Try a bigger model.")
        words = [w for s in segs for w in s["words"]]
        clips = []
        for i, c in enumerate(picks, 1):
            log(f"Rendering clip {i} of {len(picks)}", 50 + 40 * (i - 1) // len(picks))
            ws = [w for w in words if w["s"] >= c["start"] - 0.05 and w["e"] <= c["end"] + 0.3]
            render(src, c, ws, d, i, f.get("layout", "crop"))
            clips.append({"file": f"clip{i}.mp4", "title": c.get("title", ""), "score": c["score"],
                          "reason": c.get("reason", ""), "start": c["start"], "end": c["end"]})
        log("Writing posts in your voice", 92)
        try:
            posts = write_posts(cfg, segs, f.get("voice", ""))
        except Exception:
            posts = None
        j["result"] = {"clips": clips, "posts": posts}
        (d / "result.json").write_text(json.dumps(j["result"]))
        j["status"] = "done"; j["pct"] = 100
    except Exception as e:
        j["status"], j["error"] = "error", str(e)

# ---------- web ----------
@app.post("/api/start")
def start():
    f, jid = request.form, uuid.uuid4().hex[:8]
    d = ROOT / jid
    d.mkdir(parents=True, exist_ok=True)
    src, up = None, request.files.get("file")
    cookie_up = request.files.get("cookies")
    cookies_upload = None
    if cookie_up and cookie_up.filename:
        try:
            cookies_upload = cookie_up.read().decode("utf-8")
        except UnicodeDecodeError:
            return jsonify(error="The cookies file must be a UTF-8 Netscape cookies.txt file"), 400
    if up and up.filename:
        src = d / ("source" + Path(up.filename).suffix)
        up.save(src)
    elif not f.get("url"):
        return jsonify(error="Paste a link or choose a file"), 400
    job_form = dict(f)
    job_form["cookies_upload"] = cookies_upload
    JOBS[jid] = {"status": "running", "pct": 0, "log": [], "result": None, "error": None}
    threading.Thread(target=run, args=(jid, job_form, src), daemon=True).start()
    return jsonify(id=jid)

@app.get("/api/job/<jid>")
def job(jid):
    if jid in JOBS:
        return jsonify(JOBS[jid])
    r = ROOT / jid / "result.json"
    if r.exists():
        return jsonify(status="done", pct=100, log=[], result=json.loads(r.read_text()))
    return jsonify(error="not found"), 404

@app.get("/files/<jid>/<name>")
def files(jid, name):
    return send_from_directory(ROOT / jid, name)

@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Clipmaker</title>
<style>
:root{--bg:#f3f5fb;--s:#fff;--ink:#14183a;--mute:#5a5f80;--line:#dde1f0;--ac:#3d5afe;--aci:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#0e1124;--s:#171b38;--ink:#eef0ff;--mute:#a3a8cc;--line:#2a2f57;--ac:#7b8cff;--aci:#0e1124}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 18px 80px}h1{font-size:34px;letter-spacing:-.03em;margin:8px 0}h2{margin:32px 0 12px}
.card,form{background:var(--s);border:1px solid var(--line);border-radius:16px;padding:16px}
input,select,textarea{width:100%;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit}
label{display:block;font-size:14px;color:var(--mute);margin:12px 0 4px}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
button,.btn{background:var(--ac);color:var(--aci);border:0;border-radius:999px;padding:10px 20px;font:600 16px inherit;cursor:pointer;margin-top:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}video{width:100%;border-radius:10px;background:#000;aspect-ratio:9/16}
.mute{color:var(--mute);font-size:14px}.card p{white-space:pre-wrap;margin:6px 0}.card a{color:var(--ac)}
#bar{height:8px;background:var(--ac);border-radius:8px;width:0;transition:width .4s}#track{background:var(--line);border-radius:8px;margin:8px 0}
</style></head><body><main>
<h1>Clipmaker</h1><p class="mute">One video in. Captioned vertical clips and posts out. Upload a file directly, or paste a public video link.</p>
<form id="f">
<label>Video link (YouTube, TikTok, Instagram, X, Facebook)</label><input name="url" type="url" placeholder="https://...">
<label>or upload a file</label><input name="file" type="file" accept="video/*,audio/*">
<label>YouTube cookies (optional, only if YouTube asks you to sign in)</label><input name="cookies" type="file" accept=".txt,text/plain">
<label>Browser User-Agent (optional, used with YouTube cookies)</label><input name="user_agent" type="text" placeholder="Mozilla/5.0 ...">
<label>Your own posts (a few, so the writing sounds like you)</label><textarea name="voice" rows="4"></textarea>
<div class="row">
<div><label>Clips</label><input name="n" type="number" min="1" max="10" value="5"></div>
<div><label>Layout</label><select name="layout"><option value="crop">Crop to fill</option><option value="fit">Fit on blurred background</option></select></div>
<div><label>Whisper model</label><select name="whisper"><option>tiny</option><option>base</option><option selected>small</option><option>medium</option></select></div>
<div><label>AI</label><select name="provider" id="prov"><option value="ollama">Ollama (local)</option><option value="anthropic">Claude API</option></select></div>
<div><label>Model</label><input name="model" id="model" value="llama3.1"></div>
</div>
<button>Make clips</button></form>
<div id="prog" hidden><div id="track"><div id="bar"></div></div><div id="log" class="mute"></div></div>
<div id="res"></div>
<script>
const $=s=>document.querySelector(s);
const esc=t=>String(t==null?"":t).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("#prov").onchange=e=>$("#model").value=e.target.value=="anthropic"?"claude-sonnet-5-5":"llama3.1";
$("#f").onsubmit=async e=>{e.preventDefault();
 const d=await (await fetch("/api/start",{method:"POST",body:new FormData(e.target)})).json();
 if(d.error)return alert(d.error);location.hash=d.id;watch(d.id)};
function watch(id){$("#prog").hidden=false;$("#res").innerHTML="";
 const t=setInterval(async()=>{const j=await (await fetch("/api/job/"+id)).json();
  $("#bar").style.width=(j.pct||0)+"%";$("#log").textContent=(j.log||[]).slice(-1)[0]||"";
  if(j.status=="done"){clearInterval(t);show(id,j.result)}
  if(j.status=="error"||j.error){clearInterval(t);$("#log").textContent="Failed: "+(j.error||"unknown")}},1500)}
function show(id,r){$("#bar").style.width="100%";$("#log").textContent="Done";
 let h="<h2>Clips</h2><div class=grid>"+r.clips.map(c=>`<div class=card><video controls preload=metadata src="/files/${id}/${c.file}"></video><b>${esc(c.title)}</b><div class=mute>Score ${c.score}. ${esc(c.reason)}</div><a href="/files/${id}/${c.file}" download>Download</a></div>`).join("")+"</div>";
 const p=r.posts;
 if(p){h+="<h2>Posts</h2>"+[...(p.x_thread||[]).map((t,i)=>["X post "+(i+1),t]),["LinkedIn",p.linkedin],["Newsletter",p.newsletter]].map(([k,t])=>`<div class=card style="margin-bottom:12px"><b>${k}</b><p>${esc(t)}</p><button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent)">Copy</button></div>`).join("")}
 else h+="<p class=mute>The posts step failed, but your clips are ready.</p>";
 $("#res").innerHTML=h}
if(location.hash.length>1)watch(location.hash.slice(1));
</script></main></body></html>"""

if __name__ == "__main__":
    print("Open http://localhost:5000")
    app.run(host="127.0.0.1", port=5000)