/* @ts-self-types="./clankers_dsp.d.ts" */

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClankersBassFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_clankersbass_free(ptr, 0);
    }
    /**
     * @param {number} seed
     */
    constructor(seed) {
        const ret = wasm.clankersbass_new(seed);
        this.__wbg_ptr = ret >>> 0;
        ClankersBassFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Render n_samples of audio (adds all active voices). Returns Float32Array.
     * @param {number} n_samples
     * @returns {Float32Array}
     */
    render(n_samples) {
        const ret = wasm.clankersbass_render(this.__wbg_ptr, n_samples);
        return ret;
    }
    /**
     * Trigger a note. cc_json: '{"74":80,"71":60}' or '{}'.
     * hold_samples: note-on duration in samples (0 = use amp envelope only)
     * @param {number} midi_note
     * @param {number} velocity
     * @param {number} hold_samples
     * @param {string} cc_json
     */
    trigger(midi_note, velocity, hold_samples, cc_json) {
        const ptr0 = passStringToWasm0(cc_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.clankersbass_trigger(this.__wbg_ptr, midi_note, velocity, hold_samples, ptr0, len0);
    }
    /**
     * Trigger + render full tail — isolated single voice, no shared state.
     * Note: ClankerBoy uses MIDI 0-23 for bass roots. We transpose +24 semitones
     * so the actual synthesis sits in the audible 50-200 Hz range.
     * @param {number} midi_note
     * @param {number} velocity
     * @param {number} hold_samples
     * @param {string} cc_json
     * @returns {Float32Array}
     */
    trigger_render(midi_note, velocity, hold_samples, cc_json) {
        const ptr0 = passStringToWasm0(cc_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.clankersbass_trigger_render(this.__wbg_ptr, midi_note, velocity, hold_samples, ptr0, len0);
        return ret;
    }
}
if (Symbol.dispose) ClankersBass.prototype[Symbol.dispose] = ClankersBass.prototype.free;

/**
 * Buchla 259/292 — percussive LPG arp with FM + wavefolding (8 voices).
 *
 * ClankerBoy CC map (t:1):
 *   CC74 cutoff  CC71 resonance  CC20 wavefold  CC17 fm_depth
 *   CC18 fm_index  CC19 env_decay  CC16 volume
 */
export class ClankersBuchla {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClankersBuchlaFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_clankersbuchla_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.clankersbuchla_new();
        this.__wbg_ptr = ret >>> 0;
        ClankersBuchlaFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Trigger + render full tail — isolated single voice.
     * @param {number} midi_note
     * @param {number} velocity
     * @param {string} cc_json
     * @returns {Float32Array}
     */
    trigger_render(midi_note, velocity, cc_json) {
        const ptr0 = passStringToWasm0(cc_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.clankersbuchla_trigger_render(this.__wbg_ptr, midi_note, velocity, ptr0, len0);
        return ret;
    }
}
if (Symbol.dispose) ClankersBuchla.prototype[Symbol.dispose] = ClankersBuchla.prototype.free;

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClankersDrumsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_clankersdrums_free(ptr, 0);
    }
    /**
     * @param {number} seed
     */
    constructor(seed) {
        const ret = wasm.clankersdrums_new(seed);
        this.__wbg_ptr = ret >>> 0;
        ClankersDrumsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Trigger a hit and immediately render its full tail.
     * Uses an isolated voice — no shared engine state contamination.
     * @param {number} voice_id
     * @param {number} velocity
     * @param {number} p0
     * @param {number} p1
     * @param {number} p2
     * @returns {Float32Array}
     */
    trigger_render(voice_id, velocity, p0, p1, p2) {
        const ret = wasm.clankersdrums_trigger_render(this.__wbg_ptr, voice_id, velocity, p0, p1, p2);
        return ret;
    }
}
if (Symbol.dispose) ClankersDrums.prototype[Symbol.dispose] = ClankersDrums.prototype.free;

/**
 * HybridSynth pads — Moog ladder + ADSR + chorus + reverb (8 polyphonic voices).
 *
 * trigger_render(midi_note, velocity, hold_samples, cc_json) → stereo Float32Array
 * hold_samples: note-on duration in samples (beat * 60/bpm * 44100)
 * Returns interleaved stereo [L0, R0, L1, R1, ...]
 */
export class ClankersPads {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ClankersPadsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_clankerspads_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.clankerspads_new();
        this.__wbg_ptr = ret >>> 0;
        ClankersPadsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} midi_note
     * @param {number} velocity
     * @param {number} hold_samples
     * @param {string} cc_json
     * @returns {Float32Array}
     */
    trigger_render(midi_note, velocity, hold_samples, cc_json) {
        const ptr0 = passStringToWasm0(cc_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.clankerspads_trigger_render(this.__wbg_ptr, midi_note, velocity, hold_samples, ptr0, len0);
        return ret;
    }
}
if (Symbol.dispose) ClankersPads.prototype[Symbol.dispose] = ClankersPads.prototype.free;

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_6ddd609b62940d55: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_new_from_slice_ff2c15e8e05ffdfc: function(arg0, arg1) {
            const ret = new Float32Array(getArrayF32FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_81c1c31d4432cb9f: function(arg0) {
            const ret = new Float32Array(arg0 >>> 0);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./clankers_dsp_bg.js": import0,
    };
}

const ClankersBassFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_clankersbass_free(ptr >>> 0, 1));
const ClankersBuchlaFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_clankersbuchla_free(ptr >>> 0, 1));
const ClankersDrumsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_clankersdrums_free(ptr >>> 0, 1));
const ClankersPadsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_clankerspads_free(ptr >>> 0, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('clankers_dsp_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
