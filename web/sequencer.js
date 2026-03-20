/**
 * ClankerBoy JSON Step Sequencer — Web Audio lookahead scheduler
 *
 * Reads a ClankerBoy JSON sheet and drives WASM drum engine triggers
 * with sample-accurate scheduling via AudioContext.currentTime.
 *
 * Usage:
 *   const seq = new Sequencer(audioCtx, drumsEngine);
 *   seq.load(sheet);   // ClankerBoy JSON object
 *   seq.start();
 *   seq.stop();
 */

const LOOKAHEAD_MS  = 100;   // schedule this far ahead
const INTERVAL_MS   = 25;    // scheduler poll interval
const DRUMS_TRACK   = 10;    // t:10 = AntigravityDrums

export class Sequencer {
  constructor(ctx, drumsEngine) {
    this.ctx    = ctx;
    this.drums  = drumsEngine;
    this.sheet  = null;
    this._timer = null;

    // Playback state
    this._bpm        = 120;
    this._steps      = [];    // flat array of { beatTime, voiceId, velocity, p0, p1, p2 }
    this._loopBeats  = 0;
    this._startTime  = 0;     // ctx.currentTime when playback began
    this._nextBeat   = 0;     // beat-clock cursor (in beats)
    this._stepIdx    = 0;     // index into _steps for the next un-scheduled step
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  load(sheet) {
    this.sheet = sheet;
    this._bpm  = sheet.bpm ?? 120;
    this._compile(sheet);
  }

  start() {
    if (!this.sheet) throw new Error('No sheet loaded');
    if (this._timer) return; // already running

    this._startTime = this.ctx.currentTime + 0.05; // tiny lead-in
    this._nextBeat  = 0;
    this._stepIdx   = 0;
    this._timer = setInterval(() => this._tick(), INTERVAL_MS);
    this._tick(); // fire immediately so first beat isn't late
  }

  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  }

  get isPlaying() { return this._timer !== null; }

  // ── Internal ────────────────────────────────────────────────────────────────

  /** Flatten sheet.steps[] into a sorted list of { beatTime, voiceId, … } */
  _compile(sheet) {
    const events = [];
    let beat = 0;

    for (const step of sheet.steps ?? []) {
      const d = step.d ?? 0.5;

      for (const track of step.tracks ?? []) {
        if (track.t !== DRUMS_TRACK) continue;
        const notes = track.n ?? [];
        const vel   = (track.v ?? 100) / 127;

        for (const note of notes) {
          const { voiceId, p0, p1, p2 } = drumNoteToParams(note, track);
          events.push({ beatTime: beat, voiceId, velocity: vel, p0, p1, p2 });
        }
      }

      beat += d;
    }

    this._steps     = events.sort((a, b) => a.beatTime - b.beatTime);
    this._loopBeats = beat;

    console.log(`[seq] compiled ${events.length} drum events over ${beat} beats @ ${this._bpm} BPM`);
  }

  _beatsToSeconds(beats) {
    return beats * (60 / this._bpm);
  }

  _tick() {
    if (!this.sheet) return;

    const lookaheadSec = LOOKAHEAD_MS / 1000;
    const scheduleUntil = this.ctx.currentTime + lookaheadSec;

    // Schedule all events that fall within the lookahead window
    while (true) {
      // Wrap step index
      if (this._steps.length === 0) break;
      if (this._stepIdx >= this._steps.length) {
        // Loop: advance the beat cursor by one full loop
        this._nextBeat += this._loopBeats;
        this._stepIdx   = 0;
      }

      const ev       = this._steps[this._stepIdx];
      const evBeat   = this._nextBeat - (this._nextBeat % this._loopBeats) + ev.beatTime;
      const evTime   = this._startTime + this._beatsToSeconds(evBeat);

      if (evTime > scheduleUntil) break; // not yet

      this._scheduleHit(ev, evTime);
      this._stepIdx++;
    }
  }

  _scheduleHit(ev, when) {
    // Render samples in WASM (sync, cheap — pre-rendered buffer)
    const samples     = this.drums.trigger_render(ev.voiceId, ev.velocity, ev.p0, ev.p1, ev.p2);
    const audioBuffer = this.ctx.createBuffer(1, samples.length, this.ctx.sampleRate);
    audioBuffer.copyToChannel(samples, 0);

    const src = this.ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(this.ctx.destination);
    src.start(when);
  }
}

// ── Drum note → voice / params mapping ────────────────────────────────────────

/**
 * Maps ClankerBoy drum MIDI notes (t:10) to AntigravityDrums voice IDs.
 * Voice IDs: 0=Kick 1=Snare 2=HH Closed 3=HH Open 4=Tom L 5=Tom M 6=Tom H
 */
function drumNoteToParams(note, track) {
  const cc = track.cc ?? {};

  if (note === 36) {
    // Kick — CC74=pitch, CC23=sweep, CC75=decay (reuse available CCs)
    return {
      voiceId: 0,
      p0: norm(cc[74], 0, 127, 0.45),  // pitch
      p1: norm(cc[23], 0, 127, 0.30),  // sweep
      p2: norm(cc[75], 0, 127, 0.50),  // decay
    };
  }
  if (note === 38 || note === 40) {
    // Snare / rimshot
    return {
      voiceId: 1,
      p0: norm(cc[74], 0, 127, 0.40),  // pitch
      p1: norm(cc[75], 0, 127, 0.50),  // decay
      p2: norm(cc[71], 0, 127, 0.20),  // resonance
    };
  }
  if (note === 42 || note === 49 || note === 50 || note === 51 || note === 52 || note === 53) {
    // Closed HH family
    return {
      voiceId: 2,
      p0: norm(cc[74], 0, 127, 0.24),
      p1: norm(cc[74], 0, 127, 0.60),
      p2: 0.1,
    };
  }
  if (note === 46 || note === 54 || note === 55 || note === 56 || note === 57) {
    // Open HH family
    return {
      voiceId: 3,
      p0: norm(cc[74], 0, 127, 0.30),
      p1: norm(cc[74], 0, 127, 0.65),
      p2: 0.1,
    };
  }
  if (note === 41 || note === 43) {
    return { voiceId: 4, p0: 0.3, p1: 0.5, p2: 0.5 }; // Tom L
  }
  if (note === 45 || note === 47) {
    return { voiceId: 5, p0: 0.5, p1: 0.5, p2: 0.5 }; // Tom M
  }
  if (note === 48 || note === 50) {
    return { voiceId: 6, p0: 0.7, p1: 0.5, p2: 0.5 }; // Tom H
  }

  // Default: kick
  return { voiceId: 0, p0: 0.45, p1: 0.30, p2: 0.50 };
}

/** Normalise a CC value (0–127) to 0–1. Returns defaultVal if cc is undefined. */
function norm(val, min, max, defaultVal) {
  if (val === undefined || val === null) return defaultVal;
  return (val - min) / (max - min);
}
