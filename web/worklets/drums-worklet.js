/**
 * Drums AudioWorkletProcessor — Clankers 3
 * Uses no-modules wasm-bindgen bundle via importScripts.
 *
 * Messages from main thread:
 *   { type: 'trigger', voiceId, velocity, p0, p1, p2 }
 */

importScripts('/wasm/clankers_dsp.js');

let engine = null;

class DrumsWorkletProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();

        const wasmBytes = options?.processorOptions?.wasmBytes;

        // wasm_bindgen is set on globalThis by the patched no-modules bundle
        wasm_bindgen(wasmBytes).then(() => {
            const seed = (Math.random() * 0xffffffff) >>> 0;
            engine = new wasm_bindgen.ClankersDrums(seed);
            this.port.postMessage({ type: 'ready' });
        }).catch(e => {
            this.port.postMessage({ type: 'error', message: String(e) });
        });

        this.port.onmessage = (e) => {
            if (!engine) return;
            const { type, voiceId, velocity, p0 = 0.5, p1 = 0.5, p2 = 0.5 } = e.data;
            if (type === 'trigger') {
                engine.trigger(voiceId, velocity, p0, p1, p2);
            }
        };
    }

    process(_inputs, outputs) {
        const out = outputs[0][0];
        if (!engine || !out) return true;
        const buf = engine.process(out.length);
        out.set(buf);
        return true;
    }
}

registerProcessor('drums-worklet', DrumsWorkletProcessor);
