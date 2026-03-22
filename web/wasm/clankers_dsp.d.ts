/* tslint:disable */
/* eslint-disable */

/**
 * Pro-One style polyphonic bass (8 voices, TPT ladder filter).
 *
 * ClankerBoy CC map (all normalised 0-127):
 *   CC74 cutoff  CC71 resonance  CC73 amp_attack  CC75 amp_decay
 *   CC79 amp_sustain  CC72 amp_release  CC23 flt_decay  CC18 detune_cents
 *   CC5  glide_time
 *
 * trigger(midi_note, velocity_0_1, cc_json_string)
 *   cc_json_string: JSON object of CC values, e.g. '{"74":80,"71":60}'
 *
 * render(n_samples) → Float32Array  (call after trigger, before next trigger)
 */
export class ClankersBass {
    free(): void;
    [Symbol.dispose](): void;
    constructor(seed: number);
    /**
     * Render n_samples of audio (adds all active voices). Returns Float32Array.
     */
    render(n_samples: number): Float32Array;
    /**
     * Trigger a note. cc_json: '{"74":80,"71":60}' or '{}'.
     * hold_samples: note-on duration in samples (0 = use amp envelope only)
     */
    trigger(midi_note: number, velocity: number, hold_samples: number, cc_json: string): void;
    /**
     * Trigger + render full tail — isolated single voice, no shared state.
     * Note: ClankerBoy uses MIDI 0-23 for bass roots. We transpose +24 semitones
     * so the actual synthesis sits in the audible 50-200 Hz range.
     */
    trigger_render(midi_note: number, velocity: number, hold_samples: number, cc_json: string): Float32Array;
}

/**
 * Buchla 259/292 — percussive LPG arp with FM + wavefolding (8 voices).
 *
 * ClankerBoy CC map (t:1):
 *   CC74 cutoff  CC71 resonance  CC20 wavefold  CC17 fm_depth
 *   CC18 fm_index  CC19 env_decay  CC16 volume
 */
export class ClankersBuchla {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    /**
     * Trigger + render full tail — isolated single voice.
     */
    trigger_render(midi_note: number, velocity: number, cc_json: string): Float32Array;
}

/**
 * Voice IDs:  0=Kick  1=Snare  2=HiHat Closed  3=HiHat Open
 *             4=Tom L  5=Tom M  6=Tom H
 *
 * Trigger params (p0..p2):
 *   Kick:   p0=pitch(0-1)  p1=sweep_time(0-1)  p2=decay(0-1)
 *   Snare:  p0=pitch(0-1)  p1=decay(0-1)        p2=resonance(0-1)
 *   HiHat:  p0=decay(0-1)  p1=cutoff(0-1)       p2=resonance(0-1)
 *   Tom:    p0=pitch(0-1)  p1=decay(0-1)         p2=unused
 */
export class ClankersDrums {
    free(): void;
    [Symbol.dispose](): void;
    constructor(seed: number);
    /**
     * Trigger a hit and immediately render its full tail.
     * Uses an isolated voice — no shared engine state contamination.
     */
    trigger_render(voice_id: number, velocity: number, p0: number, p1: number, p2: number): Float32Array;
}

/**
 * HybridSynth pads — Moog ladder + ADSR + chorus + reverb (8 polyphonic voices).
 *
 * trigger_render(midi_note, velocity, hold_samples, cc_json) → stereo Float32Array
 * hold_samples: note-on duration in samples (beat * 60/bpm * 44100)
 * Returns interleaved stereo [L0, R0, L1, R1, ...]
 */
export class ClankersPads {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    trigger_render(midi_note: number, velocity: number, hold_samples: number, cc_json: string): Float32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_clankersbass_free: (a: number, b: number) => void;
    readonly __wbg_clankersbuchla_free: (a: number, b: number) => void;
    readonly __wbg_clankersdrums_free: (a: number, b: number) => void;
    readonly __wbg_clankerspads_free: (a: number, b: number) => void;
    readonly clankersbass_new: (a: number) => number;
    readonly clankersbass_render: (a: number, b: number) => any;
    readonly clankersbass_trigger: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly clankersbass_trigger_render: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly clankersbuchla_new: () => number;
    readonly clankersbuchla_trigger_render: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly clankersdrums_new: (a: number) => number;
    readonly clankersdrums_trigger_render: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly clankerspads_new: () => number;
    readonly clankerspads_trigger_render: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
