# Clankers3 Web Architecture Plan

> **Status**: Planning — no implementation yet.
> **Goal**: Transform Clankers3 from a Python CLI pipeline into a live, browser-native generative music companion with real-time synthesis and chat control.

---

## 1. Current Architecture (Baseline)

```
Brief (text)
    ↓
[Chatroom]  Claude + Gemini + ChatGPT negotiate → Music Sheet JSON
    ↓
[Conductor] For each section in arc:
    ├── evolve(sheet, section) via GPT-4o
    ├── run_session() — parallel agents:
    │   ├── bass_sh101  → LLM sequence + NumPy DSP → WAV
    │   ├── drums       → LLM 16-step + NumPy DSP → WAV
    │   ├── harmony     → LLM progression + Buchla + Hybrid/Granular → WAV×2
    │   └── voder       → LLM phonemes + formant synth → WAV
    ├── mixer.py — EQ + gain per track
    └── stitch + master → full_track.wav
```

**What renders audio**: Python + NumPy DSP, optionally DawDreamer VST3 backends.
**What the Music Sheet is**: An intermediate format — created, consumed, then discarded once the WAV is rendered.

---

## 2. Target Architecture

```
Browser                                         Python Backend (FastAPI)
───────────────────────────────                 ──────────────────────────
  Chat UI ─── user types ────────────────────▶  POST /chat
                  ◀────── updated Music Sheet ──  (LLM call, no audio)
  Sequencer reads Music Sheet
  AudioWorkletProcessors play live              POST /session/new
  Parameter knobs ─► worklet messages           POST /sheet/evolve
  Music loops continuously ◀─── loop event      GET  /sheet/{id}
```

**What changes**:
- The Music Sheet becomes the **persistent living state** of a composition, not a render step.
- All audio synthesis moves to the **browser** (Rust → WASM AudioWorklets).
- The Python backend becomes a **thin JSON API** — LLM calls and sheet management only.
- Real-time parameter changes hit the running worklet directly — no re-render, no API round-trip.

---

## 3. Chosen DSP Strategy: Rust → WASM AudioWorklets

**Why Rust over pure JS**: The voder formant engine and granular processor are numerically intensive. Rust → WASM gives near-native performance with predictable timing, avoids GC pauses inside the audio callback, and ports the existing NumPy math directly (types are clear, no implicit broadcasting to worry about).

**Why not Pyodide**: ~20 MB download, slow startup, unpredictable GC in audio callbacks. Ruled out.

**Why not pure JS**: Viable for drums and bass but the voder's 5-formant bank with coarticulation, the Clouds granular with grain scheduling, and the PolyBLEP oscillators all benefit from Rust's determinism. Consistent toolchain across all instruments also matters.

### Rust crate structure (proposed)

```
clankers_dsp/          (Rust workspace)
├── Cargo.toml
├── src/
│   ├── lib.rs         — wasm_bindgen exports
│   ├── common/
│   │   ├── oscillators.rs    (square, sawtooth PolyBLEP, sub-osc)
│   │   ├── filters.rs        (biquad, one-pole, Butterworth cascade)
│   │   ├── envelopes.rs      (ADSR, accent, smoothing)
│   │   └── effects.rs        (chorus, reverb/wash, overdrive, distortion)
│   ├── bass/
│   │   └── sh101.rs          (Pro-One voice: square+sub, filter, envelope, vibrato)
│   ├── drums/
│   │   └── engine.rs         (kick, snare, hihat×2, clap synthesis)
│   ├── harmony/
│   │   ├── buchla.rs         (triangle, wavefold, FM, fast envelope)
│   │   └── hybrid.rs         (detuned saws, granular/Clouds, slow envelope)
│   └── voder/
│       ├── engine.rs         (formant bank, glottal pulse, coarticulation)
│       ├── phoneme_table.rs  (30 phonemes, F1-F5 + bandwidths + gains)
│       └── plosives.rs       (closure / burst / aspiration phases)
```

Each instrument exposes a `process(buffer: &mut [f32], params: &InstrumentParams)` function callable from JS via `wasm_bindgen`.

---

## 4. Backend Changes (FastAPI)

The Python backend is stripped of all audio synthesis. It becomes:

### 4.1 Endpoints

```
POST /session/new
  body: { brief: str, arc?: str[], agents?: str[] }
  → { session_id: str, sheet: MusicSheet }
  # agents: if provided, only those companions start active (solo/subset mode)
  # e.g. agents: ["harmony"] → only the buchla/hybrid companion plays

POST /chat
  body: { session_id: str, message: str, target_agent?: str }
  → { sheet: MusicSheet, diff: SheetDiff, reply: str }

POST /sheet/evolve
  body: { session_id: str, section: str }
  → { sheet: MusicSheet }

GET  /sheet/{session_id}
  → { sheet: MusicSheet }

PATCH /sheet/{session_id}
  body: Partial<MusicSheet>   (user direct edits)
  → { sheet: MusicSheet }
```

### 4.2 Agent changes

Each agent's `run()` function currently: **calls LLM → synthesizes audio → exports WAV**.

After the split, each agent exports two functions:

```python
# agents/bassline/bass_sh101.py

def generate_sequence(sheet: dict, api_key: str) -> dict:
    """LLM call only. Returns note sequence + synth params."""
    # unchanged — already exists

def render_sequence(notes, bpm, ...) -> AudioSegment:
    """NumPy DSP. Keep for offline/testing, not called by web server."""
    # unchanged — preserved for CLI path
```

The API server calls only `generate_sequence()`. The rendered audio path stays intact for CLI use and testing.

### 4.3 Session state

Sessions are held in memory (dict keyed by UUID) during development. Each session holds:

```python
@dataclass
class Session:
    id: str
    sheet: dict              # current Music Sheet JSON
    section_history: list    # previous section sheets
    chat_history: list       # LLM conversation log
    created_at: float
```

### 4.4 What stays Python-only

- `chatroom.py` — multi-LLM negotiation (unchanged)
- `conductor.py` — evolve logic (unchanged, minus audio dispatch)
- `mixer.py` — only used by CLI render path now
- All `generate_sequence()` functions in each agent (unchanged)
- All `render_*()` / `synth_*()` functions (kept, just not called by web server)

---

## 5. Music Sheet as Living State

The Music Sheet JSON is the single source of truth. It does not change structurally — every field already maps to a browser parameter:

| Sheet field | Worklet parameter |
|---|---|
| `agents.bass_sh101.synth.filter_cutoff` | `Sh101Worklet.filterCutoff` (0–1 → 300–4000 Hz) |
| `agents.bass_sh101.synth.pulse_width` | `Sh101Worklet.pulseWidth` (0.25–0.5) |
| `agents.drums.synth.kick.pitch_hz` | `DrumsWorklet.kickPitch` (40–90 Hz) |
| `agents.drums.synth.kick.decay_s` | `DrumsWorklet.kickDecay` |
| `agents.harmony.synth.hybrid.cloud.density` | `HybridWorklet.grainDensity` (0–1 → 2–60 /s) |
| `agents.harmony.synth.hybrid.cloud.freeze` | `HybridWorklet.freeze` (bool) |
| `agents.voder.fundamental_hz` | `VoderWorklet.fundamentalHz` |
| ... | ... |

When the user turns a knob, it sends a `postMessage` to the running worklet — the sheet is also updated locally so it stays in sync. When the LLM responds to a chat message it returns an updated sheet; the browser diffs it and sends only changed params to worklets.

---

## 6. JavaScript Sequencer

Lives in the browser. Reads from the Music Sheet, drives all worklets.

```
AudioContext.currentTime  →  scheduler (lookahead 100ms, interval 25ms)
    ↓
Bass sequencer:
  reads sheet.agents.bass_sh101 notes array
  sends note-on/note-off messages to Sh101Worklet
  handles slide (portamento flag), accent (velocity boost)

Drums sequencer:
  reads 16-step arrays per voice
  fires on 16th-note grid (with swing offset from sheet)
  velocity → worklet hit-strength param

Harmony sequencer:
  reads chord progression + arpeggio pattern
  dispatches note events to BuchlaWorklet and HybridWorklet

Voder sequencer:
  reads word timeline (phonemes, note, duration)
  triggers phoneme transitions in VoderWorklet
```

Key constraint: **all scheduling uses `AudioContext.currentTime`**, never `setTimeout`/`setInterval` for note timing. The Web Audio clock is sample-accurate; JS timers are not.

---

## 7. Companion Chat UI

Each agent maps to a companion persona with a name and visual identity. The chat UI shows which companion is responding.

| Companion | Agent | Role |
|---|---|---|
| **The Bassist** | bass_sh101 | Warm, dry, musical — talks about feel and groove |
| **The Drummer** | drums | Terse, rhythmic — talks about energy and patterns |
| **Keys** | harmony (buchla + hybrid) | Harmonic, opinionated — textures and progressions |
| **The Voice** | voder | Mysterious — phonemes, breath, formant space |
| **The Conductor** | chatroom/conductor | Orchestrates the others; listens to user intent |

**Solo invocation**: each companion tile is independently clickable. Clicking a dormant companion with a brief invokes it alone — only its worklet starts, the others remain silent. The user hears just that one voice. Other companions can be invited in at any time by clicking their tile or addressing them in chat.

User messages route as follows:
1. User types in the chat box.
2. Frontend sends `POST /chat` with the message and session ID.
3. Backend uses the Conductor to determine which agents are addressed (or all).
4. Each addressed agent's `generate_sequence()` is called with the updated sheet.
5. Backend returns the updated sheet + a companion reply string.
6. Frontend diffs the sheet, pushes updated params to running worklets.
7. Music responds on the next loop without stopping.

Example interactions:
- *"Make the kick heavier"* → Drummer updates `kick.pitch_hz` and `kick.punch`
- *"Darker chords, more tension"* → Keys updates chord progression and `pad_cutoff`
- *"Something feels off with the groove"* → Conductor re-evaluates global notes, may sync bass and kick
- *"Sing something menacing"* → Voice updates phoneme hints and `fundamental_hz`
- Clicking the **Keys** tile with a brief → `POST /session/new { agents: ["harmony"] }` → only buchla and hybrid play

---

## 8. Frontend Structure

```
web/
├── index.html
├── main.js              — app bootstrap, AudioContext init
├── session.js           — session management, sheet state, API calls
├── sequencer.js         — timing engine, note scheduling
├── companions/
│   ├── chat.js          — chat UI, message routing, companion personas
│   └── roster.js        — companion tiles: click to invoke solo, mute, or invite into band
├── ui/
│   ├── knobs.js         — parameter knob components
│   ├── sheet-view.js    — visual display of Music Sheet (optional debug panel)
│   └── status.js        — session status, section indicator
├── worklets/
│   ├── sh101-worklet.js
│   ├── drums-worklet.js
│   ├── buchla-worklet.js
│   ├── hybrid-worklet.js
│   └── voder-worklet.js
└── wasm/
    └── clankers_dsp_bg.wasm   (compiled output, loaded by worklets)
```

**`companions/roster.js`** renders one tile per companion. Each tile has three states:
- **dormant** — agent `active: false`, worklet loaded but silent
- **solo** — agent `active: true`, all others `active: false`; clicking this tile from idle starts a solo session
- **in band** — agent `active: true` alongside others

Clicking a dormant tile with no active session opens a brief input for that companion; submitting it calls `POST /session/new` with `agents: [that_agent]`. Clicking while a session is running toggles the agent in/out of the mix via `PATCH /sheet/{id}`.

Each worklet file is an `AudioWorkletProcessor` subclass:

```js
// worklets/drums-worklet.js
import init, { DrumsEngine } from '../wasm/clankers_dsp.js';

class DrumsWorklet extends AudioWorkletProcessor {
  async init() {
    await init();
    this.engine = new DrumsEngine(sampleRate);
  }
  process(inputs, outputs, parameters) {
    this.engine.process(outputs[0][0]);
    return true;
  }
}
registerProcessor('drums-worklet', DrumsWorklet);
```

---

## 9. Parameter Smoothing Strategy

Abrupt parameter changes cause audible zipper noise. The DSP layer handles this:

- **Rust**: Use an `OnePoleSmooth` (exponential moving average) on every modulatable param. Port from `voder/dsp.py:OnePoleSmooth`. Time constant ~5ms for fast params (filter), ~50ms for slow ones (reverb depth).
- **JS knob → worklet**: Send `postMessage({type: 'param', name, value})`. Worklet updates smooth target, not raw value.
- **LLM sheet update → worklet**: Same message path. Changes land gradually.

---

## 10. Implementation Sequence

The recommended order minimises wasted work and validates each layer before building on it.

### Phase 0 — Note known improvements first

> *Before touching the web architecture, capture any improvements to how instruments produce music or how the DSP works (per user's note). These changes stay in the Python DSP layer and inform what the Rust port needs to match.*

- Finalize bass_sh101 synthesis improvements
- Finalize drums engine changes
- Finalize harmony (buchla / hybrid / granular) changes
- Finalize voder formant changes
- Run CLI `session.py` end-to-end to confirm output quality

### Phase 1 — Backend split

1. Create `api/` directory, `api/main.py` (FastAPI app)
2. Add `api/session_store.py` (in-memory session dict)
3. Add `api/routes/session.py` — `POST /session/new` (supports `agents?: str[]` for solo/subset mode)
4. Add `api/routes/chat.py` — `POST /chat` (calls conductor/chatroom, returns sheet only)
5. Add `api/routes/sheet.py` — `GET`, `PATCH`, `POST /sheet/evolve`
6. Verify: `curl POST /session/new` with `{"brief": "dreamy buchla", "agents": ["harmony"]}` returns a sheet with only harmony active
7. Verify: `curl POST /chat` updates the sheet without rendering audio

### Phase 2 — Rust DSP crate (start with drums)

1. `cargo init clankers_dsp`; add `wasm-bindgen`, `wasm-pack`
2. Port `drums_agent.py` DSP functions to `src/drums/engine.rs`
   - `synth_kick(pitch_hz, punch, decay_s)` → `&[f32]`
   - `synth_snare(tuning_hz, snappy)` → `&[f32]`
   - `synth_hihat(decay_ms, highpass_hz, open: bool)` → `&[f32]`
   - `synth_clap()` → `&[f32]`
3. Expose `DrumsEngine::process(buf: &mut [f32])` via wasm_bindgen
4. `wasm-pack build --target web`
5. Write `web/worklets/drums-worklet.js` loading the WASM
6. Smoke test: standalone HTML page plays a kick on click

### Phase 3 — Sequencer (drums only)

1. Write `web/sequencer.js` with 16-step drum grid driven by `AudioContext.currentTime`
2. Read `sheet.agents.drums.pattern` arrays
3. Apply swing from `sheet.agents.drums.synth` (if present)
4. Verify timing accuracy at 120 BPM over 8 bars

### Phase 4 — End-to-end proof of concept

1. Minimal `index.html` with companion tiles (one per agent) and a brief input
2. Clicking the **Drummer** tile + brief → `POST /session/new { agents: ["drums"] }` → only drums worklet starts (solo mode)
3. Verify music plays
4. Add one knob: **Kick Pitch** (range 40–90 Hz)
5. Verify knob sends `postMessage` to worklet and pitch changes in real time without stopping
6. Add **invite** button on other tiles — clicking activates that agent and patches the sheet

### Phase 5 — Remaining Rust instruments

Port in this order (complexity ascending):
1. **Bass (Sh101)** — square+sub oscillators, 4-pole LP, envelope, vibrato, portamento
2. **Buchla** — triangle→wavefold, FM, fast envelope
3. **HybridSynth** — detuned saws, granular/Clouds (most complex in this group)
4. **Voder** — formant bank, glottal pulse, coarticulation, 30-phoneme table (most complex overall)

For each:
- Port Rust DSP
- Write JS worklet
- Add sequencer logic for that instrument
- Add parameter knobs to UI

### Phase 6 — Companion chat UI

1. Build chat panel in `web/companions/chat.js`
2. Connect to `POST /chat` endpoint
3. Display companion name + reply text per response
4. Diff returned sheet against current sheet
5. Push changed params to worklets via postMessage
6. Add visual indicator when music is updating
7. Build `companions/roster.js` — companion tiles with dormant / solo / in-band states
8. Wire solo invocation: tile click → brief input → `POST /session/new { agents: [name] }`
9. Wire invite: clicking a dormant tile during playback → `PATCH /sheet` to activate that agent

### Phase 7 — Polish and session management

- Section arc UI (show current section, trigger evolve)
- Save/load sessions (export sheet JSON)
- Mobile-friendly layout
- Worklet error handling and reconnect
- CORS config for production

---

## 11. File-level Change Summary

### Files unchanged by web work
- All `generate_sequence()` functions in all agents
- `chatroom/chatroom.py`
- `conductor/conductor.py` (evolve logic)
- `mixer/mixer.py` (CLI path only)
- `llm_clients.py`
- `config.py`
- `session.py` (CLI entry point, stays working)

### Files modified
- `agents/*/` — each `run()` function becomes a thin wrapper that calls `generate_sequence()` only (audio path behind an `if render_audio:` flag so CLI still works)

### New files
- `api/` — FastAPI application
- `clankers_dsp/` — Rust WASM crate
- `web/` — Frontend

---

## 12. Key Design Principles

1. **The Music Sheet is the interface.** Every change (LLM, user knob, section evolution) writes to the sheet. The sequencer reads from it. Nothing bypasses this.

2. **The CLI path must keep working.** Python DSP is not deleted. The backend API calls `generate_sequence()` only; the full `run()` path remains for offline rendering and testing.

3. **No zipper noise.** Every modulatable parameter uses `OnePoleSmooth` in Rust. All UI-to-worklet communication uses the smooth target, not the raw value.

4. **Scheduling is clock-based, not timer-based.** All note triggers are scheduled against `AudioContext.currentTime`. The JS sequencer runs on a 25ms interval but schedules 100ms ahead — this is the standard Web Audio lookahead pattern.

5. **Companions respond to the current loop.** Chat changes take effect on the next phrase boundary (bar 1 of the next loop). The sequencer checks for pending sheet updates at the top of each loop.

---

## 13. Open Questions (to resolve before implementation)

- **Session persistence**: In-memory for now. Add Redis or SQLite if sessions need to survive server restarts.
- **Multi-user**: Each browser session gets its own AudioContext and session ID. No server-side audio, so this is trivially concurrent.
- **Voder timing model**: The current Python voder schedules phonemes at the sample level. The JS/Rust port needs a message-passing model (scheduler sends phoneme events to the worklet ahead of time, similar to note scheduling).
- **Granular (Clouds) buffer**: The Clouds processor needs an audio buffer to granulate. In Python it granulates the pad's own rendered audio. In the browser it will granulate a circular buffer of the live hybrid pad output. Architecture TBD.
- **VST path**: DawDreamer VST3 paths are desktop-only. These are simply not exposed in the web API. The Rust DSP replaces them in the browser.

---

*Last updated: 2026-03-20*
*Branch: `claude/hay-chat-integration-bSrf2`*
