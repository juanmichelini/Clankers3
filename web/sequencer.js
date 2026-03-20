/**
 * ClankerBoy JSON Step Sequencer — Web Audio lookahead scheduler
 *
 * Reads a ClankerBoy JSON sheet and drives WASM instrument engines.
 * Supported tracks:
 *   t:10  AntigravityDrums (ClankersDrums)
 *   t:2   Pro-One Bass     (ClankersBass)
 *
 * Usage:
 *   const seq = new Sequencer(audioCtx, { drums, bass });
 *   seq.load(sheet);
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
    this.sheet  = null;
    this._timer = null;

    this._bpm       = 120;
    this._steps     = [];   // [{ beatTime, track, ... }]
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
    const events = [];
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
            events.push({ beatTime: beat, type: 'drum', voiceId, velocity: vel, p0, p1, p2 });
          }
        }

        if (track.t === 2 && this.bass) {
          for (const note of notes) {
            const ccJson = JSON.stringify(cc);
            events.push({ beatTime: beat, type: 'bass', midiNote: note, velocity: vel, ccJson });
          }
        }

        if (track.t === 1 && this.buchla) {
          for (const note of notes) {
            const ccJson = JSON.stringify(cc);
            events.push({ beatTime: beat, type: 'buchla', midiNote: note, velocity: vel, ccJson });
          }
        }
      }

      beat += d;
    }

    this._steps     = events.sort((a, b) => a.beatTime - b.beatTime);
    this._loopBeats = beat;
    console.log(`[seq] compiled ${events.length} events (${beat} beats @ ${this._bpm} BPM)`);
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

      this._scheduleEvent(ev, evTime);
      this._stepIdx++;
    }
  }

  _scheduleEvent(ev, when) {
    if (ev.type === 'drum') {
      const samples = this.drums.trigger_render(ev.voiceId, ev.velocity, ev.p0, ev.p1, ev.p2);
      this._playBuffer(samples, when);
    } else if (ev.type === 'bass') {
      const samples = this.bass.trigger_render(ev.midiNote, ev.velocity, ev.ccJson);
      this._playBuffer(samples, when);
    } else if (ev.type === 'buchla') {
      const samples = this.buchla.trigger_render(ev.midiNote, ev.velocity, ev.ccJson);
      this._playBuffer(samples, when);
    }
  }

  _playBuffer(samples, when) {
    if (!samples || samples.length === 0) return;
    const ab = this.ctx.createBuffer(1, samples.length, this.ctx.sampleRate);
    ab.copyToChannel(samples, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = ab;
    src.connect(this.ctx.destination);
    src.start(when);
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
