// GridCompute LiteRT.js Web Worker — Distributed ML Inference
// Runs .tflite model inference using LiteRT.js (WebGPU accelerated, XNNPACK CPU fallback).
// Integrates with the existing GridCompute socket protocol via 'process_task' / 'task_completed'.
//
// Supported task types:
//   'litert'  — run a .tflite model on provided input tensors (LiteRT.js)
//   'custom'  — generic async JS (pass-through, same as worker.js)
//
// Node registration:
//   Browser nodes using this worker should include { litert: true } in their
//   register_node profile so the server routes inference tasks to them.

'use strict';

// ── Runtime state ─────────────────────────────────────────────────────────────
let liteRtInitialized = false;
let liteRtInitPromise  = null;
const modelCache       = new Map();   // modelUrl → { model, accelerator }

// LiteRT.js API references (populated after init)
let LiteRT_loadAndCompile = null;
let LiteRT_Tensor         = null;

// Active accelerator (reported back with results for diagnostics)
let activeAccelerator = 'unknown';

// ── Initialise LiteRT.js runtime (lazy, once) ─────────────────────────────────
// Attempts to load from the locally served bundle (/litert-js/),
// then falls back to unpkg CDN. Initialises Wasm with wasm assets at /litert-wasm/.
async function initLiteRt() {
    if (liteRtInitialized) return;
    if (liteRtInitPromise)  return liteRtInitPromise;

    liteRtInitPromise = (async () => {
        // ── Step 1: load the JS bundle ────────────────────────────────────────
        const localBundle = '/litert-js/litertjs.umd.js';
        const cdnBundle   = 'https://unpkg.com/@litertjs/core/dist/litertjs.umd.min.js';

        let bundleLoaded = false;
        try {
            importScripts(localBundle);
            bundleLoaded = true;
        } catch (_) {
            // local bundle not available — try CDN
        }

        if (!bundleLoaded) {
            try {
                importScripts(cdnBundle);
                bundleLoaded = true;
            } catch (e) {
                throw new Error(
                    'LiteRT.js bundle not available locally or via CDN. ' +
                    'Run `npm install @litertjs/core` on the server. ' +
                    `CDN error: ${e.message}`
                );
            }
        }

        // ── Step 2: locate the UMD global ─────────────────────────────────────
        // The UMD build may export as LiteRT, litertjs, or the scoped pkg name.
        const LiteRTModule =
            self.LiteRT       ||
            self.litertjs     ||
            self['@litertjs/core'];

        if (!LiteRTModule || typeof LiteRTModule.loadLiteRt !== 'function') {
            throw new Error(
                'LiteRT.js UMD global not found after importScripts. ' +
                'Expected self.LiteRT with loadLiteRt / loadAndCompile / Tensor exports.'
            );
        }

        // ── Step 3: initialise the Wasm runtime ───────────────────────────────
        // /litert-wasm/ is served by the GridCompute server from
        // node_modules/@litertjs/core/wasm/
        await LiteRTModule.loadLiteRt('/litert-wasm/');

        LiteRT_loadAndCompile = LiteRTModule.loadAndCompile;
        LiteRT_Tensor         = LiteRTModule.Tensor;
        liteRtInitialized     = true;

        console.log('[LiteRT Worker] Runtime initialised successfully.');
    })();

    return liteRtInitPromise;
}

// ── Load + compile a .tflite model (cached by URL + accelerator) ──────────────
async function getCompiledModel(modelUrl) {
    const cacheKey = modelUrl;
    if (modelCache.has(cacheKey)) return modelCache.get(cacheKey);

    let model;

    // Try WebGPU first (ML Drift engine — 5–60× faster than CPU for large models)
    try {
        model = await LiteRT_loadAndCompile(modelUrl, { accelerator: 'webgpu' });
        activeAccelerator = 'webgpu';
        console.log(`[LiteRT Worker] ✅ Model compiled (WebGPU): ${modelUrl}`);
    } catch (gpuErr) {
        console.warn(
            `[LiteRT Worker] ⚠️ WebGPU unavailable (${gpuErr.message}). ` +
            'Falling back to XNNPACK CPU…'
        );

        // Fall back to XNNPACK (CPU via Wasm — wide browser support)
        try {
            model = await LiteRT_loadAndCompile(modelUrl, { accelerator: 'xnnpack' });
            activeAccelerator = 'xnnpack';
            console.log(`[LiteRT Worker] ✅ Model compiled (XNNPACK/CPU): ${modelUrl}`);
        } catch (cpuErr) {
            throw new Error(
                `Model compilation failed. ` +
                `WebGPU: ${gpuErr.message} | XNNPACK: ${cpuErr.message}`
            );
        }
    }

    modelCache.set(cacheKey, model);
    return model;
}

// ── Build a typed array from the task input descriptor ────────────────────────
function buildTypedArray(data, dtype) {
    switch ((dtype || 'float32').toLowerCase()) {
        case 'int32':  return new Int32Array(data);
        case 'uint8':  return new Uint8Array(data);
        case 'bool':   return new Uint8Array(data);
        default:       return new Float32Array(data);  // float32
    }
}

// ── Main message handler ──────────────────────────────────────────────────────
self.onmessage = async function (e) {
    const task = e.data;

    // ── 1. Custom JS pass-through (same as worker.js) ─────────────────────────
    if (task.type === 'custom') {
        const AsyncFn = Object.getPrototypeOf(async function () {}).constructor;
        try {
            const fn     = new AsyncFn('params', task.script);
            const result = await fn(task.params);
            self.postMessage({ jobId: task.jobId, taskId: task.taskId, result });
        } catch (err) {
            self.postMessage({
                jobId:  task.jobId,
                taskId: task.taskId,
                error:  err.message || 'Custom task execution error'
            });
        }
        return;
    }

    // ── 2. LiteRT inference task ───────────────────────────────────────────────
    if (task.type === 'litert') {
        const { jobId, taskId, modelUrl, inputs } = task;

        // Validate
        if (!modelUrl || typeof modelUrl !== 'string') {
            self.postMessage({ jobId, taskId, error: 'litert task missing modelUrl' });
            return;
        }
        if (!Array.isArray(inputs) || inputs.length === 0) {
            self.postMessage({ jobId, taskId, error: 'litert task missing inputs array' });
            return;
        }

        try {
            // Lazy-boot the LiteRT runtime (one-time cost)
            await initLiteRt();

            // Load + compile model (cached after first use)
            const model = await getCompiledModel(modelUrl);

            // Build input tensor(s)
            const inputTensors = inputs.map(({ data, shape, dtype }) => {
                if (!Array.isArray(data) || !Array.isArray(shape)) {
                    throw new Error('Each input must have data (array) and shape (array)');
                }
                const typedData = buildTypedArray(data, dtype);
                return new LiteRT_Tensor(typedData, shape);
            });

            // ── Run inference ─────────────────────────────────────────────────
            const t0        = performance.now();
            const rawOutputs = await model.run(...inputTensors);
            const inferenceMs = parseFloat((performance.now() - t0).toFixed(2));

            // Move GPU tensors → Wasm memory for JS-side serialisation
            const outputs = await Promise.all(
                rawOutputs.map(async (tensor) => {
                    const cpuTensor = await tensor.moveTo('wasm');
                    return {
                        data:  Array.from(cpuTensor.toTypedArray()),
                        shape: Array.from(cpuTensor.shape),
                        dtype: cpuTensor.dtype || 'float32'
                    };
                })
            );

            self.postMessage({
                jobId,
                taskId,
                result: {
                    outputs,
                    inferenceMs,
                    modelUrl,
                    accelerator: activeAccelerator
                }
            });

        } catch (err) {
            self.postMessage({
                jobId,
                taskId,
                error: err.message || 'LiteRT inference failed'
            });
        }
        return;
    }

    // ── 3. Unsupported task type ───────────────────────────────────────────────
    self.postMessage({
        jobId:  task?.jobId,
        taskId: task?.taskId,
        error:  `worker_litert.js: unsupported task type "${task?.type}". ` +
                `Supported: "litert", "custom".`
    });
};
