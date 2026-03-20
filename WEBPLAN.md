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

---

## 14. Genre-Reactive Visual Identity

The page and companions change their entire look based on the style of music being made. The Music Sheet's `mood` field, combined with the user's brief, is mapped to a `genre_tag` by the Conductor. This tag drives two visual layers: the **page theme** and the **companion skins**.

### 14.1 Genre Tag

The Conductor derives a `genre_tag` from the brief and mood when a session starts (or when a major style shift occurs via chat). It is stored in the Music Sheet:

```json
{
  "genre_tag": "punk_synthwave",
  "mood": "dark, cold, aggressive",
  ...
}
```

Supported genre families:

| `genre_tag` | Example briefs |
|---|---|
| `punk_synthwave` | "dark industrial EBM", "cold acid punk", "neon-soaked aggression" |
| `jazz` | "late-night jazz club", "cool bossa", "bebop with electronics" |
| `ambient` | "drone, floating", "slow shimmer", "meditative pads" |
| `industrial_ebm` | "mechanical, brutal, body music", "factory floor percussion" |
| `acid_house` | "acid rave", "four-to-the-floor party", "303 squelch party" |
| `classical` | "orchestral tension", "neo-classical electronics", "chamber synth" |
| `reggae_dub` | "heavy dub space", "roots reggae", "echoed bass and percussion" |
| `hip_hop` | "lo-fi boom bap", "trap synth", "dusty samples and 808s" |

### 14.2 Page Themes

Each genre tag maps to a CSS theme class applied to `<body>`. Themes control:

- Background color, texture, and animated overlays
- Accent colors (neon, warm, cool, muted)
- Typography weight and letter-spacing
- Border/divider style (hard grid, soft blur, organic)
- Subtle background animations (scanlines, grain, floating particles, slow fog)

```css
/* web/themes/punk-synthwave.css */
body.punk-synthwave {
  --bg: #0a0010;
  --surface: #12001e;
  --accent-1: #ff2d78;    /* hot pink neon */
  --accent-2: #00f5ff;    /* cyan neon */
  --text: #e0e0ff;
  --font-weight: 900;
  --border: 2px solid var(--accent-1);
  background-image: url('textures/scanlines.svg');
  animation: scanline-scroll 8s linear infinite;
}

body.jazz {
  --bg: #1a1008;
  --surface: #261808;
  --accent-1: #d4a843;    /* warm amber */
  --accent-2: #8fbcbb;    /* cool teal */
  --text: #f0e6d0;
  --font-weight: 400;
  --border: 1px solid rgba(212, 168, 67, 0.4);
  background-image: url('textures/grain.svg');
}

body.ambient {
  --bg: #080c14;
  --surface: #0c1020;
  --accent-1: #7aa2d4;    /* soft blue */
  --accent-2: #a0c4a0;    /* muted green */
  --font-weight: 300;
  --border: 1px solid rgba(122, 162, 212, 0.2);
  /* slow radial gradient pulse animation */
}
```

Theme files live in `web/themes/`. The page transitions on genre change via a short crossfade (300ms opacity on the `<body>` background and a companion "wardrobe swap" animation).

### 14.3 Companion Skins

Each companion has a base visual design (SVG or CSS illustration) with swappable **clothing/accessory layers** per genre. The base personality and face stay recognizable — only the look changes.

| Companion | Base look | punk_synthwave | jazz | ambient | industrial_ebm | acid_house |
|---|---|---|---|---|---|---|
| The Bassist | Laid-back, warm | Fingerless gloves, leather jacket, neon bass | Low-lit bar slouch, upright bass shadow | Barefoot, loose shirt, eyes half-closed | Hard hat, boots, heavy instrument rig | Bucket hat, glowstick necklace |
| The Drummer | Compact, energetic | Mohawk sprite, torn tee | Brushes visible, beret | Soft mallets, calm posture | Industrial goggles, rivet-gun drumsticks | Massive sunglasses, hands blurred |
| Keys | Opinionated, gesturing | Synth worn like a guitar, cyber visor | Turtleneck, cigarette holder, bar stool | Flowing sleeves, softly glowing keys | Welding mask pushed up, heavy gloves | Oversized smiley tee, keyboard like a surfboard |
| The Voice | Mysterious, minimal | Neon half-mask, face obscured | Vintage microphone, spotlight glow | Facing slightly away, half in shadow | Industrial respirator partially removed | Whistle around neck, wide eyes |
| The Conductor | Formal but warm | Ripped tuxedo jacket, synth clipboard | Proper conductor's baton, tails | No baton, just gentle hand gestures | Clipboard replaced by punch-card printout | Clipboard is a glowstick wand |

Skins are implemented as layered SVGs or CSS classes swapped on the companion tile element. Each companion tile has:

```html
<div class="companion-tile" data-agent="drums" data-genre="punk_synthwave">
  <div class="companion-sprite">
    <img class="base" src="sprites/drums/base.svg" />
    <img class="costume" src="sprites/drums/punk_synthwave.svg" />
    <img class="accessory" src="sprites/drums/punk_synthwave_accessory.svg" />
  </div>
  <span class="companion-name">The Drummer</span>
  <span class="companion-state">in band</span>
</div>
```

### 14.4 Personality Tone Shifts

The companion personalities are fixed, but their *register* shifts with the genre. The backend injects the genre tag into the system prompt for each companion's LLM call:

```python
GENRE_TONE = {
    "punk_synthwave": "You're in punk mode. Be blunt, a little aggressive, impatient — but still helpful and never mean.",
    "jazz": "You're in jazz mode. Be measured, cool, slightly mysterious. Economical with words.",
    "ambient": "You're in ambient mode. Slow down. Be gentle, spacious, almost sleepy. Long pauses are fine.",
    "industrial_ebm": "You're in industrial mode. Mechanical, precise, serious. Every word counts.",
    "acid_house": "You're in acid house mode. Enthusiastic, chaotic, fast. Lots of energy.",
    "classical": "You're in classical mode. Formal, precise, slightly imperious. But kind.",
}
```

Examples of the same companion across genres:

> **The Drummer on kick drum, punk_synthwave:**
> "That kick is too soft. Punch it up. Decay shorter. More pitch drop."

> **The Drummer on kick drum, jazz:**
> "Let's keep the kick understated — just enough to anchor the ride."

> **The Drummer on kick drum, ambient:**
> "Maybe... no kick at all? Or very far away. Like a heartbeat you almost imagine."

### 14.5 Genre Transitions

When a chat message signals a major style shift ("make it more jazzy", "go darker"), the Conductor updates `genre_tag` in the sheet. The frontend detects the change in the sheet diff and:

1. Fades the page background to the new theme (300ms)
2. Triggers companion "wardrobe swap" animation (sprites fade/spin briefly)
3. The chat log shows a small visual separator: *"— style shift: jazz —"*

---

## 15. Schooling Mode

Every companion can switch into a teacher role, explaining music theory, instrument design, and synthesis concepts in real time. The goal is that someone with zero music production knowledge can build a track while learning exactly what every parameter does and why it sounds the way it does.

### 15.1 Triggering School Mode

Three ways to activate:

1. **Global toggle** — A `?` button in the UI header enables School Mode globally. All companions become narrators of their own decisions.
2. **Per-companion** — A small `?` badge on each companion tile activates their teaching mode individually. Only that companion explains their work.
3. **Natural language in chat** — Companions recognise and respond to:
   - "How does that work?"
   - "Teach me about [topic]"
   - "Why did you do that?"
   - "What is [parameter]?"
   - "Explain [synthesis concept]"

### 15.2 What Each Companion Teaches

**The Drummer** — Rhythm and drum synthesis:
- What a 16-step grid is and how it maps to bars and beats
- Swing: what it does, why 54% feels groove-ier than 50%
- How a kick drum is synthesised (sine chirp + noise burst + pitch envelope)
- Snare: tuned noise + transient, why "snappy" controls the high-frequency tail
- Hi-hat: filtered noise, why open vs closed is just decay length
- Polyrhythm: two patterns of different lengths running simultaneously
- The difference between four-on-the-floor, breakbeat, and half-time

**The Bassist** — Basslines and filter synthesis:
- Scale degrees: why bass uses degrees (1, 3, 5) instead of note names
- Octave register: why bass stays in octave 1–2
- What portamento (slide) does acoustically and why it sounds smooth
- Filter cutoff: the "curtain" metaphor — what frequencies pass through at each value
- Resonance: why a peak at the cutoff frequency creates that acid squelch
- Pulse width: how a square wave changes shape and why wider sounds thicker
- Sub oscillator: adding a fundamental one octave below for weight

**Keys** — Harmony and synthesis:
- Chord construction: root, third, fifth, seventh — why they work together
- Tension and release: diminished/augmented chords vs resolved chords
- Why a slow attack makes a pad feel like it's "fading in from nothing"
- FM synthesis: the carrier/modulator relationship, FM index as brightness
- Wavefolding: what happens when you push a waveform past its limits (soft clipping vs folding)
- Granular synthesis: reading tiny frozen grains of a sound out of order and cross-fading them
- What "cloud density" controls: how many grains per second are playing

**The Voice** — Formant synthesis and the human voice:
- What a formant is: a resonant peak in the vocal tract
- How vowels differ: "ah" vs "ee" is F1/F2 frequency shift, not pitch
- Glottal pulse: the vibration of the vocal folds is the source; the vocal tract shapes it
- Coarticulation: why the mouth starts moving for the next phoneme before the current one ends
- What "fundamental_hz" controls: pitch, not vowel colour
- Why whispering works: noise source instead of glottal pulse, same formants
- The difference between voiced (b, d, z) and unvoiced (p, t, s) consonants

**The Conductor** — Music structure and production:
- What a "section" is and why tracks change over time (verse/chorus/bridge)
- Energy arc: why most tracks build toward a peak and release
- Why the Music Sheet has a `globalNotes` field: inter-agent coordination ("bass and kick lock on beat 1")
- What EQ does: frequency-selective volume — cut vs boost, shelf vs bell
- Compression: why loud moments get quieter and quiet ones get louder
- Why mastering chains work: saturation, limiting, loudness perception

### 15.3 Real-Time Annotations

In School Mode, when a companion makes a change, their chat reply includes a brief explanation appended after the musical decision:

```
[Keys] → I've widened the chord voicing and pulled the pad cutoff down to 0.3.

  📖 The cutoff (0.3) removes frequencies above roughly 900 Hz, leaving only the
  warm low-mid body of the pad. A low cutoff on a slow-attack pad creates that
  "emerging from fog" effect — you hear the fundamental before any brightness appears.
```

The annotation is visually distinguished (lighter color, slightly smaller text, a book emoji prefix).

### 15.4 Interactive Lessons

Companions can run short structured lessons triggered by the user:

- `"Teach me about kick drums"` → The Drummer walks through kick synthesis step by step, adjusting the kick parameters live as they explain each stage
- `"Show me what resonance does"` → The Bassist sweeps the resonance parameter slowly while explaining the effect
- `"What is an arpeggio?"` → Keys sets up a simple chord and plays it first block, then arpeggiated, explaining the difference
- `"How do vowels work?"` → The Voice cycles through ah / ee / oh while explaining which formants shifted and why

Lessons are structured as a short sequence of sheet mutations + companion narration messages, delivered at a reading pace (one step every few seconds). The music plays live throughout.

### 15.5 Backend Support

School Mode adds a flag to the session and chat API:

```json
POST /session/new
{
  "brief": "...",
  "school_mode": true,
  "school_focus": "drums"   // optional: only that companion teaches
}

POST /chat
{
  "session_id": "...",
  "message": "teach me about the filter",
  "school_mode": true
}
```

The backend injects a teaching instruction into the system prompt when school mode is active:

```python
SCHOOL_MODE_SUFFIX = """
When school_mode is active, append a brief educational note after your musical decision.
Use plain language. Explain ONE concept per response. Max 3 sentences.
Prefix the note with 📖.
"""
```

### 15.6 Visual Support in School Mode

- The sheet-view panel (optional debug panel in Section 8) becomes a **live annotated parameter display**: every field shows its current value, its human-readable range, and a one-line description of what it does.
- Active parameters (ones that just changed) highlight briefly with a pulse animation.
- A mini waveform or spectrum display appears per companion, showing the raw output of their synth engine in real time.

### 15.7 Genre Skin in School Mode

School Mode companions wear a slightly different visual: a pair of small glasses appears on each sprite (a subtle "teacher mode" accessory), and the page gains a thin chalkboard-texture overlay in the background while the genre theme remains active underneath. Removing School Mode removes the glasses and chalkboard overlay.

---

## 16. Knowledge Progression — Unlock System

Learning is the unlock key. The more a user understands about music production, the more of the tool they can access. Genre styles, synthesis engines, companion abilities, and UI features are all locked behind knowledge milestones rather than time-gating or paywalls.

### 16.1 Core Idea

Every concept taught in Section 15 is a **learnable unit**. When the user engages with a unit (reads a lesson, asks a question, adjusts a parameter mid-lesson, or correctly answers a comprehension question), that unit is marked learned. Enough learned units in a domain unlocks the next tier of tools in that domain.

The progression is **musical and tangible** — you don't unlock a badge, you unlock a sound.

### 16.2 Knowledge Units

Each companion owns a set of knowledge units. These are the same concepts listed in Section 15.2, formalised:

| ID | Companion | Concept | Unlocks |
|----|-----------|---------|---------|
| `drum.grid` | Drummer | 16-step grid, bars, beats | Odd time signatures (5/4, 7/8) in the drum editor |
| `drum.swing` | Drummer | Swing, groove quantisation | Swing knob unlocked (previously hidden) |
| `drum.kick_synthesis` | Drummer | Kick: sine chirp + noise + pitch env | Manual kick synth editor (tune each stage) |
| `drum.polyrhythm` | Drummer | Two patterns of different lengths | Pattern length up to 32 steps, cross-rhythm layers |
| `drum.styles` | Drummer | 4-on-floor, breakbeat, half-time | Drum style picker: Jungle, Afrobeat, Latin, Drill |
| `bass.scale_degrees` | Bassist | Degrees 1, 3, 5, scale awareness | Modal basslines (Dorian, Mixolydian, Phrygian) |
| `bass.filter` | Bassist | Cutoff, resonance, acid squelch | Filter automation lane; envelope-to-filter routing |
| `bass.portamento` | Bassist | Slide / portamento | Portamento per-note control, pitch glide curves |
| `bass.sub` | Bassist | Sub oscillator | Dual-oscillator bass engine |
| `keys.chord_construction` | Keys | Root, third, fifth, seventh | Chord type picker: 9ths, 11ths, sus chords |
| `keys.tension` | Keys | Diminished, augmented, tension/release | Tension-arc automation: schedule tension over time |
| `keys.fm` | Keys | FM synthesis, carrier/modulator | FM engine for Keys companion (replaces default) |
| `keys.granular` | Keys | Granular synthesis, grains | Granular engine for Keys companion |
| `voice.formants` | Voice | Formant peaks, vowel shaping | Manual formant editor: drag F1/F2 on a plot |
| `voice.whisper` | Voice | Noise source vs glottal pulse | Whisper mode toggle |
| `voice.phonemes` | Voice | Voiced vs unvoiced consonants | Phoneme sequencer — write words as beat patterns |
| `conductor.structure` | Conductor | Sections, verse/chorus/bridge | Section manager: name and arrange song blocks |
| `conductor.compression` | Conductor | Compression, dynamic control | Per-companion compressor controls |
| `conductor.mastering` | Conductor | Mastering chain | Mastering chain panel with saturation, limiting |

### 16.3 Engagement Detection

A knowledge unit is marked **learned** when any of these happens:

1. **Passive lesson** — the companion teaches the concept during a structured lesson (Section 15.4) and the user stays through the final step.
2. **Natural question** — the user asks about the concept in chat ("what is resonance?", "how does swing work?") and the companion answers with school mode active.
3. **Active discovery** — the user manually adjusts a parameter that is annotated in School Mode. The annotation appears, they keep it toggled on for at least 5 seconds.
4. **Comprehension check** — the companion asks a short follow-up question ("what does lowering the cutoff do to the sound?") and the user gives an answer. Any non-empty answer counts; this is not a quiz, just active recall.

### 16.4 Unlock Flow

```
unit learned
    │
    ▼
[ progress bar fills in companion's knowledge tier ]
    │
    ├── tier not complete → no change except bar progress
    │
    └── tier complete
           │
           ▼
       unlock animation: companion does a small celebration,
       a new item appears in the UI (style, engine, control)
           │
           ▼
       companion says: "You've got a feel for [concept]. Try [unlocked thing]."
```

The unlock is always presented as a **musical invitation**, not a reward popup. The companion introduces the new tool as a natural next step.

### 16.5 Genre Style Unlocks

Genre styles are the most visible unlocks. The default set is small; the rest require knowledge:

| Style | Required Units |
|-------|---------------|
| Techno *(default)* | — |
| House *(default)* | — |
| Lo-fi Hip Hop | `drum.swing` + `keys.chord_construction` |
| Acid | `bass.filter` + `drum.grid` |
| Drum & Bass | `drum.polyrhythm` + `bass.sub` |
| Jungle | `drum.styles` + `bass.portamento` |
| Afrobeat | `drum.styles` + `keys.tension` |
| Ambient | `keys.granular` + `conductor.structure` |
| Jazz Fusion | `keys.chord_construction` + `keys.tension` + `bass.scale_degrees` |
| Drill | `drum.styles` + `bass.filter` + `conductor.compression` |
| Industrial | `drum.kick_synthesis` + `keys.fm` |
| A Cappella / Vocal | `voice.formants` + `voice.phonemes` + `conductor.structure` |
| Orchestral | `conductor.mastering` + `keys.chord_construction` + `conductor.structure` |

When a style is locked, it is visible in the genre picker as a greyed tile with a small lock icon and a tooltip: *"Learn [required concept] to unlock"*.

### 16.6 Progression State

State is stored per-session in the backend and optionally persisted per-user (if auth is added):

```python
@dataclass
class KnowledgeState:
    learned_units: set[str]          # e.g. {"drum.grid", "bass.filter"}
    unlocked_styles: set[str]        # derived from learned_units
    unlocked_engines: set[str]       # e.g. {"keys.fm", "keys.granular"}
    unlocked_controls: set[str]      # e.g. {"drum.swing_knob", "bass.filter_lane"}
```

The session API exposes this state so the frontend can render locks accurately:

```
GET /session/{id}/progress
→ { learned_units: [...], unlocked_styles: [...], unlocked_controls: [...] }
```

### 16.7 New Session Knowledge Seeding

When a user starts a new session with a genre brief (Section 3), the system infers which knowledge units they probably already have based on what that genre requires. A user who asks for a Drum & Bass track is not shown locks on `drum.polyrhythm` — they clearly understand the concept at some level. The session starts with those units pre-marked so the experience doesn't feel patronising.

### 16.8 Companion Awareness of Locks

Companions are aware of what is locked. If the user asks a companion to use an unlocked engine ("use the FM synth"), they respond normally. If the engine is locked, the companion explains the prerequisite instead of silently ignoring the request:

```
[Keys] → The FM engine isn't available yet — we haven't talked about carrier/modulator
relationships. Ask me to "teach me FM synthesis" and I'll walk you through it.
Once you've got it, the FM engine will open up.
```

### 16.9 Companion Evolution — They Grow With You

Companions don't stay static. As the user's `KnowledgeState` expands, each companion's **voice, vocabulary, and musical ambition** evolve to match.

Three growth stages per companion:

**Stage 1 — Beginner** *(0–4 units learned for that companion)*
- Explanations are fully analogical: "think of the cutoff like a curtain over the speakers"
- Musical decisions are conservative: safe chords, stable rhythms, obvious structures
- Companion is warm, patient, never uses jargon without explaining it first
- They reference only what the user has already seen

**Stage 2 — Developing** *(5–9 units)*
- Companion starts using correct terminology alongside analogy: "I'm pulling the resonance — that peak at the cutoff creates the squelch sound acid bass is known for"
- Musical decisions get more interesting: syncopation, modal borrowing, extended chords
- They start asking the user for input: "You've learned about tension — want me to add a diminished chord at the end of this phrase?"
- References concepts learned earlier without re-explaining them

**Stage 3 — Fluent** *(10+ units, or all units for that companion)*
- Companion communicates as a peer: brief, precise, no hand-holding unless asked
- Musical decisions are ambitious: polyrhythm, FM timbre design, structural tension arcs
- They collaborate rather than explain: "I want to try some cross-rhythm between the kick and bass — you know why this works"
- Schooling Mode annotations are still available on request but off by default at this stage

**Stage transitions are gradual.** There is no hard cutoff moment — vocabulary and complexity shift incrementally with each new unit learned. The companion never regresses; if the user unlocks granular and then doesn't use it for a while, the companion still treats them as someone who understands it.

**Companion memory of shared progress.** When the companion reaches Stage 2 or 3, they occasionally reference the learning journey: "Remember when we first talked about swing? You've got that instinct now — this groove is a lot more nuanced than your first session." This is generated from the `learned_units` history, not hand-authored.

### 16.10 Visual Progress Panel

A collapsible sidebar (or companion-detail drawer) shows the user's current knowledge state per companion:

- A small grid of concept bubbles, each filled or hollow.
- Hovering a hollow bubble shows: concept name, which control it unlocks, and a prompt like "Ask [companion] to teach you this".
- Filled bubbles show a one-line summary of what the user learned.
- A genre-unlock strip at the bottom shows all styles, greyed or coloured based on completion.

The panel is opt-in — the `?` button from Section 15.1 opens it. Users who don't want to think about progression can ignore it entirely.

---

*Last updated: 2026-03-20*
*Branch: `claude/clankers3-setup-8Wwqw`*
