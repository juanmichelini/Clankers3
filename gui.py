"""
gui.py -- The Clankers 3 Web GUI
Run:  python gui.py
Open: http://localhost:5050
"""

import io
import queue
import sys
import threading
import zipfile
from pathlib import Path

# Ensure the_clankers3 root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, Response, jsonify, request, send_file, stream_with_context, make_response

# ── Constants ─────────────────────────────────────────────────────────────────

OUT_DIR    = Path(__file__).resolve().parent / "output"
ARC_ORDER  = ["verse1", "instrumental", "verse2", "bridge", "verse3", "outro"]
AGENT_NAMES = ["sampler", "bass_sh101", "drums", "buchla", "hybrid", "voder"]
SENTINEL   = object()   # signals SSE stream to close

# ── Global state ──────────────────────────────────────────────────────────────

_log_queue    = queue.Queue()
_is_running   = False
_run_error    = None
_orig_stdout  = sys.stdout

# ── Stdout capture ────────────────────────────────────────────────────────────

class QueueWriter(io.TextIOBase):
    """Redirect print() output into a queue for SSE streaming."""

    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s)

    def flush(self):
        pass

# ── Background generation thread ──────────────────────────────────────────────

def _generation_thread(brief: str, arc: list, disable: list, single_llm: bool = False):
    global _is_running, _run_error

    # Lazy import to avoid startup errors from config.py import chain
    from conductor.conductor import run_track

    try:
        _is_running = True
        _run_error  = None

        # Drain stale queue entries from previous run
        while not _log_queue.empty():
            try:
                _log_queue.get_nowait()
            except queue.Empty:
                break

        # Capture stdout
        sys.stdout = QueueWriter(_log_queue)

        run_track(
            brief      = brief,
            arc        = arc or None,
            out_dir    = str(OUT_DIR),
            disable    = disable or None,
            single_llm = single_llm,
        )

    except Exception as e:
        _run_error = str(e)
        _log_queue.put(f"\n[ERROR] {e}\n")
    finally:
        sys.stdout   = _orig_stdout
        _is_running  = False
        _log_queue.put(SENTINEL)

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return HTML

@app.post("/generate")
def generate():
    global _is_running
    if _is_running:
        return jsonify({"error": "Generation already in progress"}), 409

    data       = request.get_json(force=True)
    brief      = (data.get("brief") or "").strip()
    arc        = [s for s in ARC_ORDER if s in (data.get("arc") or ARC_ORDER)]
    disable    = [a for a in AGENT_NAMES if a in (data.get("disable") or [])]
    single_llm = bool(data.get("single_llm", False))

    if not brief:
        return jsonify({"error": "Brief is required"}), 400
    if not arc:
        return jsonify({"error": "Select at least one arc section"}), 400

    t = threading.Thread(
        target   = _generation_thread,
        args     = (brief, arc, disable, single_llm),
        daemon   = True,
    )
    t.start()
    return jsonify({"status": "started"})

@app.get("/stream")
def stream():
    def event_stream():
        while True:
            try:
                item = _log_queue.get(timeout=30)
            except queue.Empty:
                yield "event: heartbeat\ndata: ping\n\n"
                continue

            if item is SENTINEL:
                yield "event: done\ndata: done\n\n"
                return

            # Escape SSE: replace bare newlines inside a data value
            text = str(item).replace("\r", "").rstrip("\n")
            for line in text.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype     = "text/event-stream",
        headers      = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/status")
def status():
    wav       = OUT_DIR / "full_track.wav"
    stems_dir = OUT_DIR / "stems"
    stems_ready = stems_dir.exists() and any(stems_dir.glob("*.wav"))
    return jsonify({
        "running":      _is_running,
        "ready":        wav.exists(),
        "stems_ready":  stems_ready,
        "error":        _run_error,
    })

@app.get("/download")
def download():
    wav = OUT_DIR / "full_track.wav"
    if not wav.exists():
        return jsonify({"error": "No track generated yet"}), 404
    return send_file(str(wav), as_attachment=True, download_name="full_track.wav")

@app.get("/download_stems")
def download_stems():
    stems_dir = OUT_DIR / "stems"
    if not stems_dir.exists():
        return jsonify({"error": "No stems generated yet"}), 404
    wavs = sorted(stems_dir.glob("*.wav"))
    if not wavs:
        return jsonify({"error": "Stems folder is empty"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for wav in wavs:
            zf.write(wav, wav.name)
    buf.seek(0)

    resp = make_response(buf.read())
    resp.headers["Content-Type"] = "application/zip"
    resp.headers["Content-Disposition"] = "attachment; filename=stems.zip"
    return resp

# ── HTML page ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Clankers 3</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0d0d0d;
    --surface: #141414;
    --panel:   #1a1a1a;
    --border:  #2a2a2a;
    --accent:  #c8ff00;
    --text:    #e0e0e0;
    --muted:   #555;
    --error:   #ff4444;
    --font:    'Courier New', Courier, monospace;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: baseline;
    gap: 16px;
  }

  header h1 {
    font-size: 15px;
    letter-spacing: 0.2em;
    color: var(--accent);
    font-weight: normal;
  }

  header span {
    color: var(--muted);
    font-size: 11px;
  }

  main {
    flex: 1;
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 0;
    height: calc(100vh - 49px);
  }

  /* ── Left panel: controls ── */
  #controls {
    border-right: 1px solid var(--border);
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    overflow-y: auto;
  }

  .section-label {
    font-size: 10px;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 8px;
    text-transform: uppercase;
  }

  textarea#brief {
    width: 100%;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--font);
    font-size: 12px;
    padding: 10px;
    resize: vertical;
    min-height: 110px;
    outline: none;
    transition: border-color 0.15s;
  }
  textarea#brief:focus { border-color: var(--accent); }
  textarea#brief::placeholder { color: var(--muted); }

  /* pill toggles */
  .toggle-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
  }

  .toggle-grid input[type="checkbox"],
  .toggle-grid input[type="radio"] { display: none; }

  .toggle-grid label {
    display: inline-block;
    padding: 5px 11px;
    border: 1px solid var(--border);
    border-radius: 20px;
    cursor: pointer;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: var(--muted);
    transition: all 0.12s;
    user-select: none;
  }

  .toggle-grid input:checked + label {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(200, 255, 0, 0.08);
  }

  /* generate button */
  #generate-btn {
    width: 100%;
    padding: 12px;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 4px;
    font-family: var(--font);
    font-size: 12px;
    letter-spacing: 0.15em;
    cursor: pointer;
    font-weight: bold;
    transition: opacity 0.15s;
    margin-top: auto;
  }

  #generate-btn:hover { opacity: 0.88; }

  #generate-btn.loading {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* ── Right panel: output ── */
  #output {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #log-header {
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  #log-header span {
    font-size: 10px;
    letter-spacing: 0.15em;
    color: var(--muted);
  }

  #clear-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: var(--font);
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 3px;
    cursor: pointer;
    letter-spacing: 0.1em;
  }
  #clear-btn:hover { border-color: var(--text); color: var(--text); }

  #log {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px;
    font-size: 11px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    color: #b0b0b0;
  }

  #log .accent { color: var(--accent); }
  #log .error  { color: var(--error); }

  #download-bar {
    border-top: 1px solid var(--border);
    padding: 14px 20px;
    flex-shrink: 0;
  }

  .dl-btn {
    width: 100%;
    padding: 10px;
    background: none;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font);
    font-size: 12px;
    letter-spacing: 0.12em;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.12s;
    margin-bottom: 6px;
  }
  .dl-btn:last-child { margin-bottom: 0; }
  .dl-btn:hover { background: rgba(200,255,0,0.1); }
  .dl-btn.stems {
    border-color: #888;
    color: #aaa;
    font-size: 11px;
  }
  .dl-btn.stems:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); background: rgba(200,255,0,0.05); }
  .dl-btn.stems:disabled { border-color: #333; color: #444; cursor: default; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <h1>THE CLANKERS 3</h1>
  <span>AI music generation system</span>
</header>

<main>

  <!-- ── Controls ── -->
  <section id="controls">

    <div>
      <div class="section-label">Brief</div>
      <textarea id="brief" placeholder="Describe the track... e.g. Synthwave 90 BPM, dark minor chords, pulsing bass, driving drums"></textarea>
    </div>

    <div>
      <div class="section-label">Presets</div>
      <div class="toggle-grid">
        <input type="checkbox" id="preset-static-dreams"><label for="preset-static-dreams">Static Dreams</label>
      </div>
    </div>

    <div>
      <div class="section-label">LLM Mode</div>
      <div class="toggle-grid">
        <input type="radio" name="llm-mode" id="mode-multi" value="multi" checked><label for="mode-multi">Multi-LLM</label>
        <input type="radio" name="llm-mode" id="mode-solo"  value="solo"><label for="mode-solo">Claude Solo</label>
      </div>
    </div>

    <div>
      <div class="section-label">Instruments</div>
      <div class="toggle-grid" id="instruments">
        <input type="checkbox" id="inst-sampler"   checked><label for="inst-sampler">sampler</label>
        <input type="checkbox" id="inst-bass_sh101" checked><label for="inst-bass_sh101">bass</label>
        <input type="checkbox" id="inst-drums"      checked><label for="inst-drums">drums</label>
        <input type="checkbox" id="inst-buchla"     checked><label for="inst-buchla">buchla</label>
        <input type="checkbox" id="inst-hybrid"     checked><label for="inst-hybrid">hybrid pad</label>
        <input type="checkbox" id="inst-voder"      checked><label for="inst-voder">voder</label>
      </div>
    </div>

    <div>
      <div class="section-label">Arc</div>
      <div class="toggle-grid" id="arc">
        <input type="checkbox" id="arc-verse1"        checked><label for="arc-verse1">verse1</label>
        <input type="checkbox" id="arc-instrumental"  checked><label for="arc-instrumental">instrumental</label>
        <input type="checkbox" id="arc-verse2"        checked><label for="arc-verse2">verse2</label>
        <input type="checkbox" id="arc-bridge"        checked><label for="arc-bridge">bridge</label>
        <input type="checkbox" id="arc-verse3"        checked><label for="arc-verse3">verse3</label>
        <input type="checkbox" id="arc-outro"         checked><label for="arc-outro">outro</label>
      </div>
    </div>

    <button id="generate-btn">GENERATE</button>

  </section>

  <!-- ── Output ── -->
  <section id="output">
    <div id="log-header">
      <span>OUTPUT LOG</span>
      <button id="clear-btn">CLEAR</button>
    </div>
    <pre id="log"></pre>
    <div id="download-bar" hidden>
      <button id="download-btn" class="dl-btn">DOWNLOAD  full_track.wav</button>
      <button id="stems-btn" class="dl-btn stems" disabled>DOWNLOAD  stems.zip</button>
    </div>
  </section>

</main>

<script>
  const ARC_ORDER   = ['verse1','instrumental','verse2','bridge','verse3','outro'];
  const AGENT_NAMES = ['sampler','bass_sh101','drums','buchla','hybrid','voder'];

  // ── Static Dreams preset brief ─────────────────────────────────────────────
  const STATIC_DREAMS_BRIEF =
`Dark ambient / industrial electronics. Slow to mid tempo (68–118 BPM). Sparse, mechanical pulse.

Buchla: wavefold 0.45, fm_depth 0.35, LPG mode Combo, long slow attack (env1_atk=0.8). Arpeggiate in minor pentatonic across 2–3 octaves, moderate rate. Occasional Cycle envelope for evolving drone tones.

Bass (Pro-One): root notes only, one note per bar, stay below C3, glide always on (portamento 0.6–0.8), filter cutoff low (0.3).

Pad (HybridSynth): heavy reverb (reverb_size=0.8, reverb_mix=0.7), slow attack (env1_atk=0.75), granular texture, dark tone.

Drums: ultra-sparse. Kick only on beat 1 every 1–2 bars, velocity 65–72. No snare. Closed hi-hat rarely (every 4 bars at most). Kit: Punchy or Analog.

Voder: sparse phonemic fragments every 8–16 bars. Breathy, unintelligible.

Arc tension: start 0.25, peak at 0.6 in verse2/bridge, return to 0.3 for outro. Long sections, no sudden changes. Let silence breathe.`;

  const btn         = document.getElementById('generate-btn');
  const briefEl     = document.getElementById('brief');
  const logEl       = document.getElementById('log');
  const downloadBar = document.getElementById('download-bar');
  const stemsBtn    = document.getElementById('stems-btn');

  let eventSource = null;

  // ── Generate ──────────────────────────────────────────────────────────────

  btn.addEventListener('click', async () => {
    if (btn.classList.contains('loading')) return;

    const brief = briefEl.value.trim();
    if (!brief) { appendLog('[ERROR] Enter a brief first.'); return; }

    const arc = ARC_ORDER.filter(s => document.getElementById('arc-' + s).checked);
    if (arc.length === 0) { appendLog('[ERROR] Select at least one arc section.'); return; }

    const disable = AGENT_NAMES.filter(a => !document.getElementById('inst-' + a).checked);
    // If both harmony sub-tracks are off, also add 'harmony' so the agent doesn't run at all
    if (disable.includes('buchla') && disable.includes('hybrid')) {
      disable.push('harmony');
    }

    const singleLlm = document.querySelector('input[name="llm-mode"]:checked').value === 'solo';

    // Clear log, hide download
    logEl.textContent = '';
    downloadBar.hidden = true;

    // Close existing SSE
    if (eventSource) { eventSource.close(); eventSource = null; }

    // Lock button
    btn.classList.add('loading');
    btn.textContent = singleLlm ? 'GENERATING (SOLO)...' : 'GENERATING...';

    // Start generation
    let resp;
    try {
      resp = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brief, arc, disable, single_llm: singleLlm }),
      });
    } catch (e) {
      appendLog('[ERROR] Could not reach server: ' + e.message);
      resetBtn();
      return;
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      appendLog('[ERROR] ' + (err.error || resp.statusText));
      resetBtn();
      return;
    }

    openStream();
  });

  // ── SSE stream ────────────────────────────────────────────────────────────

  function openStream() {
    eventSource = new EventSource('/stream');

    eventSource.onmessage = (e) => {
      appendLog(e.data);
    };

    eventSource.addEventListener('done', () => {
      eventSource.close();
      eventSource = null;
      resetBtn();
      downloadBar.hidden = false;
      // Check if stems are available too
      fetch('/status').then(r => r.json()).then(s => {
        stemsBtn.disabled = !s.stems_ready;
      }).catch(() => {});
    });

    eventSource.addEventListener('heartbeat', () => { /* keep-alive, no-op */ });

    eventSource.onerror = () => {
      appendLog('[stream disconnected]');
      eventSource.close();
      eventSource = null;
      resetBtn();
    };
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  function appendLog(text) {
    logEl.textContent += text + '\\n';
    logEl.scrollTop = logEl.scrollHeight;
  }

  function resetBtn() {
    btn.classList.remove('loading');
    btn.textContent = 'GENERATE';
  }

  document.getElementById('clear-btn').addEventListener('click', () => {
    logEl.textContent = '';
  });

  document.getElementById('download-btn').addEventListener('click', () => {
    window.location.href = '/download';
  });

  stemsBtn.addEventListener('click', () => {
    window.location.href = '/download_stems';
  });

  // ── Static Dreams preset toggle ───────────────────────────────────────────

  const presetSD   = document.getElementById('preset-static-dreams');
  let   savedBrief = '';   // stores the brief text before preset fills it

  presetSD.addEventListener('change', () => {
    if (presetSD.checked) {
      savedBrief      = briefEl.value;        // save whatever is currently there
      briefEl.value   = STATIC_DREAMS_BRIEF;
    } else {
      briefEl.value   = savedBrief;           // restore
      savedBrief      = '';
    }
  });

  // Auto-uncheck if the user manually edits the textarea while preset is active
  briefEl.addEventListener('input', () => {
    if (presetSD.checked && briefEl.value !== STATIC_DREAMS_BRIEF) {
      presetSD.checked = false;
      savedBrief       = '';
    }
  });

  // On load: show download buttons if outputs already exist
  (async () => {
    try {
      const s = await fetch('/status').then(r => r.json());
      if (s.ready) {
        downloadBar.hidden = false;
        stemsBtn.disabled = !s.stems_ready;
      }
    } catch (_) {}
  })();
</script>

</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("The Clankers 3 -- GUI")
    print(f"Output dir : {OUT_DIR}")
    print("Open       : http://localhost:5050")
    print()
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
