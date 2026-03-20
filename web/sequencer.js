/**
 * ClankerBoy JSON Step Sequencer — Web Audio lookahead scheduler
 *
 * All WASM rendering happens at load() time (pre-render).
 * The tick loop only schedules pre-computed AudioBuffers — no heavy work on the timer.
 *
 * Supported tracks:
 *   t:10  AntigravityDrums (ClankersDrums)
 *   t:2   Pro-One Bass     (ClankersBass)
 *   t:1   Buchla 259/292   (ClankersBuchla)
 *   t:6   HybridSynth Pads (ClankersPads)
 *
 * Usage:
 *   const seq = new Sequencer(audioCtx, { drums, bass, buchla, pads });
 *   seq.load(sheet);   // compiles + pre-renders all AudioBuffers
 *   seq.start();
 *   seq.stop();
 */

const LOOKAHEAD_MS = 100;
const INTERVAL_MS  = 25;

export class Sequencer {
  constructor(ctx, engines = {}) {
    this.ctx    = ctx;
    this.drums  = engines.drums  ?? null;
    this.bass   = engines.bass   ?? null;
    this.buchla = engines.buchla ?? null;
    this.pads   = engines.pads   ?? null;
    this.sheet  = null;
    this._timer = null;

    this._bpm       = 120;
    this._steps     = [];   // [{ beatTime, audioBuffer, stereo }]
    this._loopBeats = 0;
    this._startTime = 0;
    this._nextBeat  = 0;
    this._stepIdx   = 0;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  load(sheet) {
    this.sheet = sheet;
    this._bpm  = sheet.bpm ?? 120;
    this._compile(sheet);
  }

  start() {
    if (!this.sheet) throw new Error('No sheet loaded');
    if (this._timer) return;

    // Master gain — prevents clipping when multiple instruments sum
    if (!this._masterGain) {
      this._masterGain = this.ctx.createGain();
      this._masterGain.gain.value = 0.28;
      this._masterGain.connect(this.ctx.destination);
    }

    this._startTime = this.ctx.currentTime + 0.05;
    this._nextBeat  = 0;
    this._stepIdx   = 0;
    this._timer = setInterval(() => this._tick(), INTERVAL_MS);
    this._tick();
  }

  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  }

  get isPlaying() { return this._timer !== null; }

  // ── Internal ───────────────────────────────────────────────────────────────

  _compile(sheet) {
    const raw = [];
    let beat = 0;

    for (const step of sheet.steps ?? []) {
      const d = step.d ?? 0.5;

      for (const track of step.tracks ?? []) {
        const notes = track.n ?? [];
        const vel   = (track.v ?? 100) / 127;
        const cc    = track.cc ?? {};

        if (track.t === 10 && this.drums) {
          for (const note of notes) {
            const { voiceId, p0, p1, p2 } = drumNoteToParams(note, cc);
            raw.push({ beatTime: beat, type: 'drum', voiceId, velocity: vel, p0, p1, p2 });
          }
        }

        if (track.t === 2 && this.bass) {
          for (const note of notes) {
            raw.push({ beatTime: beat, type: 'bass', midiNote: note, velocity: vel,
                       ccJson: JSON.stringify(cc) });
          }
        }

        if (track.t === 1 && this.buchla) {
          for (const note of notes) {
            raw.push({ beatTime: beat, type: 'buchla', midiNote: note, velocity: vel,
                       ccJson: JSON.stringify(cc) });
          }
        }

        if (track.t === 6 && this.pads) {
          const durBeats = track.dur ?? step.d ?? 0.5;
          for (const note of notes) {
            raw.push({ beatTime: beat, type: 'pads', midiNote: note, velocity: vel,
                       ccJson: JSON.stringify(cc), durBeats });
          }
        }
      }

      beat += d;
    }

    raw.sort((a, b) => a.beatTime - b.beatTime);

    // Pre-render all WASM buffers now, not during the tick
    const t0 = performance.now();
    const events = raw.map(ev => {
      const audioBuffer = this._renderEvent(ev);
      return { beatTime: ev.beatTime, audioBuffer };
    });
    const elapsed = (performance.now() - t0).toFixed(0);

    this._steps     = events;
    this._loopBeats = beat;
    console.log(`[seq] compiled ${events.length} events (${beat} beats @ ${this._bpm} BPM) — pre-render ${elapsed}ms`);
  }

  /** Render one event to an AudioBuffer synchronously. Called at compile time. */
  _renderEvent(ev) {
    if (ev.type === 'drum') {
      const samples = this.drums.trigger_render(ev.voiceId, ev.velocity, ev.p0, ev.p1, ev.p2);
      return this._monoToAudioBuffer(samples);

    } else if (ev.type === 'bass') {
      const samples = this.bass.trigger_render(ev.midiNote, ev.velocity, ev.ccJson);
      return this._monoToAudioBuffer(samples);

    } else if (ev.type === 'buchla') {
      const samples = this.buchla.trigger_render(ev.midiNote, ev.velocity, ev.ccJson);
      return this._monoToAudioBuffer(samples);

    } else if (ev.type === 'pads') {
      const holdSamples = Math.round(ev.durBeats * (60 / this._bpm) * this.ctx.sampleRate);
      const interleaved = this.pads.trigger_render(ev.midiNote, ev.velocity, holdSamples, ev.ccJson);
      return this._stereoInterleavedToAudioBuffer(interleaved);
    }
    return null;
  }

  _beatsToSeconds(beats) { return beats * (60 / this._bpm); }

  _tick() {
    if (!this._steps.length) return;
    const scheduleUntil = this.ctx.currentTime + LOOKAHEAD_MS / 1000;

    while (true) {
      if (this._stepIdx >= this._steps.length) {
        this._nextBeat += this._loopBeats;
        this._stepIdx   = 0;
      }

      const ev     = this._steps[this._stepIdx];
      const loopN  = Math.floor(this._nextBeat / this._loopBeats) || 0;
      const evBeat = loopN * this._loopBeats + ev.beatTime;
      const evTime = this._startTime + this._beatsToSeconds(evBeat);

      if (evTime > scheduleUntil) break;

      if (ev.audioBuffer) {
        const src = this.ctx.createBufferSource();
        src.buffer = ev.audioBuffer;
        src.connect(this._masterGain ?? this.ctx.destination);
        src.start(evTime);
      }

      this._stepIdx++;
    }
  }

  // ── Buffer helpers ─────────────────────────────────────────────────────────

  _monoToAudioBuffer(samples) {
    if (!samples || samples.length === 0) return null;
    const ab = this.ctx.createBuffer(1, samples.length, this.ctx.sampleRate);
    ab.copyToChannel(samples, 0);
    return ab;
  }

  _stereoInterleavedToAudioBuffer(interleaved) {
    if (!interleaved || interleaved.length < 2) return null;
    const frames = Math.floor(interleaved.length / 2);
    const ab = this.ctx.createBuffer(2, frames, this.ctx.sampleRate);
    const l = new Float32Array(frames);
    const r = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      l[i] = interleaved[i * 2];
      r[i] = interleaved[i * 2 + 1];
    }
    ab.copyToChannel(l, 0);
    ab.copyToChannel(r, 1);
    return ab;
  }
}

// ── Drum note → voice / params ────────────────────────────────────────────────

function drumNoteToParams(note, cc) {
  const norm = (v, def) => v !== undefined ? v / 127 : def;

  if (note === 36) return {
    voiceId: 0,
    p0: norm(cc[74], 0.45), p1: norm(cc[23], 0.30), p2: norm(cc[75], 0.50),
  };
  if (note === 38 || note === 40) return {
    voiceId: 1,
    p0: norm(cc[74], 0.40), p1: norm(cc[75], 0.50), p2: norm(cc[71], 0.20),
  };
  if ([42, 49, 50, 51, 52, 53].includes(note)) return {
    voiceId: 2,
    p0: norm(cc[74], 0.24), p1: norm(cc[74], 0.60), p2: 0.1,
  };
  if ([46, 54, 55, 56, 57].includes(note)) return {
    voiceId: 3,
    p0: norm(cc[74], 0.30), p1: norm(cc[74], 0.65), p2: 0.1,
  };
  if (note === 41 || note === 43) return { voiceId: 4, p0: 0.3, p1: 0.5, p2: 0.5 };
  if (note === 45 || note === 47) return { voiceId: 5, p0: 0.5, p1: 0.5, p2: 0.5 };
  if (note === 48 || note === 50) return { voiceId: 6, p0: 0.7, p1: 0.5, p2: 0.5 };
  return { voiceId: 0, p0: 0.45, p1: 0.30, p2: 0.50 };
}
