/* tslint:disable */
/* eslint-disable */

/**
 * Drums engine — main-thread rendering via AudioBufferSourceNode.
 *
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
     * Returns a Float32Array ready to load into an AudioBuffer.
     * Trailing silence is trimmed so the buffer is as short as needed.
     */
    trigger_render(voice_id: number, velocity: number, p0: number, p1: number, p2: number): Float32Array;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_clankersdrums_free: (a: number, b: number) => void;
    readonly clankersdrums_new: (a: number) => number;
    readonly clankersdrums_trigger_render: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly __wbindgen_externrefs: WebAssembly.Table;
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
